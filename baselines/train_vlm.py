"""Multimodal LLM (VLM) + LoRA fine-tune as a classifier.

Architecture:
  AutoProcessor + AutoModelForCausalLM (or Vision2Seq) loaded in bf16
  + LoRA on language-model attention (q,k,v,o)
  + custom classification head on the last hidden state of the last non-pad token

Input per patient (chat format with images):
  <system>: clinical epilepsy classifier instructions
  <user>: <image-1> <image-2> ... <patient_text>  Q: <task>?

The model never has to *generate* — we directly classify from the last hidden state,
sidestepping next-token-mapping issues.

Loss: class-weighted CE on populated classes only.

Output: results/{baseline_name}/{task}_seed{seed}.json (matches existing shape)

Tested on:
  - MedGemma-4B-Multimodal (google/medgemma-4b-it)
  - Qwen2.5-VL-7B-Instruct (Qwen/Qwen2.5-VL-7B-Instruct)
"""
from __future__ import annotations
import argparse, ast, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from populated_helper import populated_ids, mask_unpopulated_logits, eval_populated, n_classes_full

SPLITS = THIS / "splits"
RESULTS = THIS / "results"
TIERS = ["silver", "gold", "bronze"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SUBFIGS = 2  # default; overridable via --max_subfigs

# Module names that LoRA should target on the VISION tower (in addition to the
# default language-tower attention list q,k,v,o). These are model-family
# specific because tower internals vary.
VISION_LORA_EXTRA = {
    "google/medgemma-4b-it":            ["out_proj"],            # SigLIP-style vision
    "Qwen/Qwen2.5-VL-7B-Instruct":      ["qkv", "proj"],         # Qwen ViT
    "Qwen/Qwen2.5-VL-32B-Instruct":     ["qkv", "proj"],
}


def _parse_paths(s) -> list[str]:
    if pd.isna(s) or s in ("", "[]"): return []
    try:
        v = ast.literal_eval(str(s))
        return [str(p) for p in v] if isinstance(v, list) else []
    except Exception:
        return []


def _build_text(r, mask_captions: bool = False) -> str:
    chunks = []
    # When mask_captions=True we drop the MRI- and EEG-report text fields, which
    # are the most likely vehicles for figure-caption-derived sentences leaking
    # into the input. Semiology, demographics_notes, age, and sex are retained
    # because they are body-narrative fields not normally populated from captions.
    field_pairs = [("semiology_text","SEMIOLOGY"), ("mri_report_text","MRI"),
                    ("eeg_report_text","EEG"), ("demographics_notes","DEMOGRAPHICS")]
    if mask_captions:
        field_pairs = [(f, l) for (f, l) in field_pairs
                        if f not in ("mri_report_text", "eeg_report_text")]
    for fld, lbl in field_pairs:
        v = r.get(fld)
        if pd.notna(v): chunks.append(f"{lbl}: {v}")
    for fld, lbl in [("age","Age"),("sex","Sex")]:
        v = r.get(fld)
        if pd.notna(v): chunks.append(f"{lbl}: {v}")
    return " | ".join(chunks)[:3000] if chunks else "[no text]"


TASK_QUESTION = {
    "epilepsy_type": "What is the patient's epilepsy_type? Choose one from: Focal, Generalised, Combined Focal and Generalised, Unknown.",
    "seizure_type": "What is the patient's seizure_type? Choose one from: Focal, Generalised, Unknown, Unclassified.",
    "ez_localization": "What is the patient's ez_localization? Choose one from: Temporal, Extratemporal, Multifocal, Hemispheric, Unknown.",
    "aed_response": "What is the patient's aed_response? Choose one from: drug-responsive, drug-resistant, unspecified.",
    "surgery_outcome": "What is the patient's surgery_outcome? Choose one from: Seizure-free, Improved.",
    "status_epilepticus": "What is the patient's status_epilepticus status? Choose one from: Non-convulsive SE, Refractory SE, Unknown.",
}


class PatientVLMDataset(Dataset):
    """Builds (chat-formatted text + images, label) tuples per patient."""
    def __init__(self, df, task, processor, max_subfigs=MAX_SUBFIGS,
                  mask_captions: bool = False):
        self.items = []
        self.processor = processor
        self.task = task
        for _, r in df.iterrows():
            mri_paths = _parse_paths(r.get("mri_image_paths"))
            eeg_paths = _parse_paths(r.get("eeg_image_paths"))
            # max_subfigs=2 keeps the original (1 MRI + 1 EEG) policy; for larger
            # caps we balance MRI + EEG and fall back to the available modality.
            if max_subfigs == 2 and mri_paths and eeg_paths:
                paths = [mri_paths[0], eeg_paths[0]]
            else:
                half = max_subfigs // 2
                if mri_paths and eeg_paths:
                    paths = (mri_paths[: max_subfigs - half] + eeg_paths[: half])[: max_subfigs]
                else:
                    paths = (mri_paths or eeg_paths)[: max_subfigs]
            text = _build_text(r, mask_captions=mask_captions)
            label = int(r[f"{task}_label_id"])
            self.items.append({"paths": paths, "text": text, "label": label})

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        # Build messages array (chat format)
        content = []
        imgs = []
        for p in it["paths"]:
            try:
                imgs.append(Image.open(p).convert("RGB").resize((448, 448)))
                content.append({"type": "image", "image": p})
            except Exception:
                pass
        content.append({"type": "text", "text": f"{it['text']}\n\n{TASK_QUESTION[self.task]}"})
        messages = [
            {"role": "system", "content": "You are a clinical epilepsy classifier."},
            {"role": "user", "content": content},
        ]
        return {"messages": messages, "images": imgs, "label": it["label"]}


def collate_vlm(batch, processor, max_length=2048):
    """Process a batch via processor.apply_chat_template + processor()."""
    texts, images_list, labels = [], [], []
    for b in batch:
        # Apply chat template to get the prompt text
        prompt = processor.apply_chat_template(
            b["messages"], tokenize=False, add_generation_prompt=True,
        )
        texts.append(prompt)
        images_list.append(b["images"] if b["images"] else None)
        labels.append(b["label"])
    # Process - some VLMs accept lists, some need single
    if all(img is None for img in images_list):
        inputs = processor(text=texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length)
    else:
        # Flatten images: pass list-of-lists
        flat_images = []
        for imgs in images_list:
            if imgs is None: imgs = []
            flat_images.append(imgs)
        inputs = processor(text=texts, images=flat_images, return_tensors="pt",
                           padding=True, truncation=True, max_length=max_length)
    inputs["labels"] = torch.tensor(labels, dtype=torch.long)
    return inputs


class VLMForClassification(nn.Module):
    def __init__(self, model_id: str, n_classes: int, dtype=torch.bfloat16,
                  vision_lora: bool = False):
        super().__init__()
        self.cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # Many VLMs have a sub-config for the language model
        self.hidden_size = (
            getattr(self.cfg, "hidden_size", None)
            or getattr(getattr(self.cfg, "text_config", None), "hidden_size", None)
            or getattr(getattr(self.cfg, "language_config", None), "hidden_size", None)
            or 4096
        )
        # Try CausalLM first (MedGemma-4B); fall back to ImageTextToText (Qwen2.5-VL,
        # Llama-3.2-Vision) then Vision2Seq for older transformers versions.
        load_kwargs = dict(dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
        last_err = None
        self.backbone = None
        for cls in [AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq]:
            if cls is None: continue
            try:
                self.backbone = cls.from_pretrained(model_id, **load_kwargs)
                print(f"  loaded backbone via {cls.__name__}")
                break
            except (ValueError, OSError, KeyError) as e:
                last_err = e
                continue
        if self.backbone is None:
            raise RuntimeError(f"could not load {model_id} via any AutoModel class: {last_err}")
        # LoRA targets: language-tower attention by default; add vision-tower
        # modules when --vision_lora is enabled (model-specific names from
        # VISION_LORA_EXTRA).
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if vision_lora:
            extras = VISION_LORA_EXTRA.get(model_id, [])
            target_modules = target_modules + extras
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=target_modules,
            task_type="FEATURE_EXTRACTION",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)
        print(f"  LoRA target_modules = {target_modules}")
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, n_classes),
        )

    def forward(self, **kwargs):
        labels = kwargs.pop("labels", None)
        # Force output_hidden_states
        kwargs["output_hidden_states"] = True
        kwargs["use_cache"] = False
        out = self.backbone(**kwargs)
        # Last hidden state of last non-pad token
        # `out.hidden_states` is a tuple of (n_layers+1, B, T, H)
        h = out.hidden_states[-1]   # (B, T, H)
        attn_mask = kwargs.get("attention_mask")
        if attn_mask is None:
            pooled = h[:, -1]
        else:
            seq_lens = attn_mask.sum(dim=1) - 1   # (B,)
            pooled = h[torch.arange(h.size(0), device=h.device), seq_lens]   # (B, H)
        logits = self.head(pooled.float())
        return logits, (F.cross_entropy(logits, labels) if labels is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True,
                    help="HF id, e.g. google/medgemma-4b-it or Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_subfigs", type=int, default=MAX_SUBFIGS,
                    help="Max image panels per patient (was hard-coded 2).")
    ap.add_argument("--vision_lora", action="store_true",
                    help="Also LoRA the vision tower (model-specific extras).")
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="Enable gradient checkpointing (needed for 32B-class).")
    ap.add_argument("--baseline_name", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--mask_captions", action="store_true",
                    help="Caption-strip ablation: drop the MRI- and EEG-report text "
                          "fields from the input prompt. Images are still passed.")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    n_classes = n_classes_full(args.task)
    pop_ids = populated_ids(args.task)
    label_col = f"{args.task}_label_id"

    print(f"=== {args.baseline_name} / {args.task} / {args.model_id} ===")
    print(f"  max_subfigs={args.max_subfigs}  vision_lora={args.vision_lora}  max_length={args.max_length}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = VLMForClassification(args.model_id, n_classes=n_classes,
                                  vision_lora=args.vision_lora).to(DEVICE)
    if args.grad_ckpt:
        try:
            model.backbone.gradient_checkpointing_enable()
            print("  gradient checkpointing enabled")
        except Exception as e:
            print(f"  [warn] grad_ckpt not enabled: {e}")
    model.backbone.print_trainable_parameters()

    train_df = pd.read_csv(SPLITS / args.task / "train.csv")
    if args.limit_train: train_df = train_df.head(args.limit_train).reset_index(drop=True)
    if args.smoke: train_df = train_df.head(20).reset_index(drop=True)

    train_ds = PatientVLMDataset(train_df, args.task, processor,
                                   max_subfigs=args.max_subfigs,
                                   mask_captions=args.mask_captions)
    if args.mask_captions:
        print(f"  [mask_captions=True] dropped MRI/EEG report text from inputs")
    print(f"  n_train={len(train_ds)}")
    if len(train_ds) < 5:
        print("[skip] too few train samples"); return

    counts = np.bincount([t["label"] for t in train_ds.items], minlength=n_classes).astype(np.float32)
    cw = np.zeros(n_classes, dtype=np.float32)
    pop_total = sum(counts[i] for i in pop_ids)
    for i in pop_ids: cw[i] = pop_total / (len(pop_ids) * max(counts[i], 1))
    cw_t = torch.from_numpy(cw).to(DEVICE)

    def _collate(batch): return collate_vlm(batch, processor, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=2, collate_fn=_collate, pin_memory=False)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=0.05)
    n_steps = max(1, (len(train_loader) // args.grad_accum) * (1 if args.smoke else args.epochs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    epochs = 1 if args.smoke else args.epochs
    print(f"  epochs={epochs} steps={n_steps} lr={args.lr} batch={args.batch}×{args.grad_accum}")
    model.train()
    step, accum = 0, 0.0
    for ep in range(epochs):
        for i, batch in enumerate(train_loader):
            batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
            labels = batch["labels"]
            inp = {k: v for k, v in batch.items() if k != "labels"}
            logits, _ = model(**inp)
            logits = mask_unpopulated_logits(logits, args.task)
            loss = F.cross_entropy(logits, labels, weight=cw_t) / args.grad_accum
            loss.backward()
            accum += loss.item() * args.grad_accum
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); sched.step(); step += 1
                if step % 5 == 0 or step == n_steps:
                    print(f"  ep {ep+1} step {step}/{n_steps} loss={accum/5:.4f}")
                    accum = 0.0
            if args.smoke and step >= 3: break
        if args.smoke: break

    # Eval
    res = {"task": args.task, "baseline": args.baseline_name, "model_id": args.model_id,
           "n_train": len(train_ds),
           "config": {"epochs": epochs, "batch": args.batch, "grad_accum": args.grad_accum,
                      "lr": args.lr, "seed": args.seed, "model": args.model_id,
                      "max_length": args.max_length, "max_subfigs": args.max_subfigs,
                      "vision_lora": args.vision_lora,
                      "populated_only": True, "lora_r": 16, "lora_alpha": 32}}
    preds_dir = RESULTS / args.baseline_name / "predictions"; preds_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for tier in TIERS:
        df_t = pd.read_csv(SPLITS / args.task / f"test_{tier}.csv")
        if args.smoke: df_t = df_t.head(20)
        ds_t = PatientVLMDataset(df_t, args.task, processor,
                                   max_subfigs=args.max_subfigs,
                                   mask_captions=args.mask_captions)
        if len(ds_t) == 0:
            res[f"macro_f1_{tier}"] = None; res[f"n_test_{tier}"] = 0; continue
        loader = DataLoader(ds_t, batch_size=args.batch, shuffle=False,
                             num_workers=2, collate_fn=_collate)
        all_logits, all_y = [], []
        with torch.no_grad():
            for b in loader:
                b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
                ys = b.pop("labels")
                logits, _ = model(**b)
                logits = mask_unpopulated_logits(logits, args.task)
                all_logits.append(logits.cpu().numpy().astype(np.float32))
                all_y.append(ys.cpu().numpy())
        logits_np = np.concatenate(all_logits, axis=0)
        y_np = np.concatenate(all_y, axis=0)
        pred = logits_np.argmax(axis=-1)
        ev = eval_populated(y_np, pred, args.task)
        res[f"macro_f1_{tier}"] = ev["macro_f1"]
        res[f"per_class_{tier}"] = ev["per_class"]
        res[f"confusion_matrix_{tier}"] = ev["confusion_matrix"]
        res[f"n_test_{tier}"] = ev["n"]
        np.savez_compressed(preds_dir / f"{args.task}_{tier}_seed{args.seed}.npz",
                              logits=logits_np, pred=pred, y=y_np)
        print(f"  {tier:>9s} n={len(ds_t)} macro-F1 (populated) = {ev['macro_f1']:.4f}")

    out = RESULTS / args.baseline_name / f"{args.task}_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
