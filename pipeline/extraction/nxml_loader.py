"""NXML/JATS parsing for v3.

Lifted from pipeline_v2/extract_patients_with_figures.py with light cleanups.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from lxml import etree


def clean_tag(tag: Any) -> str:
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def get_rich_text(node) -> str:
    if node is None:
        return ""
    parts = [node.text or ""]
    for child in node:
        parts.append(get_rich_text(child))
        parts.append(child.tail or "")
    return " ".join("".join(parts).split())


def extract_figures(root) -> dict[str, dict]:
    figures: dict[str, dict] = {}
    for fig in root.xpath(".//fig | .//fig-group/fig"):
        fig_id = fig.get("id", "")
        label = get_rich_text(fig.find("label"))
        caption = get_rich_text(fig.find("caption"))
        hrefs = [
            g.get("{http://www.w3.org/1999/xlink}href", "") or g.get("href", "")
            for g in fig.xpath(".//graphic")
        ]
        if fig_id or label:
            key = fig_id or label.replace(" ", "_")
            figures[key] = {
                "fig_id": fig_id,
                "label": label,
                "caption": caption,
                "graphic_hrefs": hrefs,
            }
    return figures


def _table_text(table_wrap) -> str:
    rows = []
    for tr in table_wrap.xpath(".//tr"):
        cells = [get_rich_text(c) for c in tr.xpath("./th | ./td")]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_tables(root) -> dict[str, dict]:
    tables: dict[str, dict] = {}
    for tw in root.xpath(".//table-wrap"):
        tid = tw.get("id", "")
        label = get_rich_text(tw.find("label"))
        caption = get_rich_text(tw.find("caption"))
        content = _table_text(tw)
        key = tid or label.replace(" ", "_") or f"table_{len(tables)+1}"
        tables[key] = {
            "table_id": tid,
            "label": label,
            "caption": caption,
            "content": content,
        }
    return tables


def _walk(element, ctx) -> str:
    buf: list[str] = []
    for child in element.iterchildren():
        if not isinstance(child.tag, str):
            continue
        tag = clean_tag(child.tag)
        if tag == "p":
            txt = get_rich_text(child)
            if txt:
                buf.append(txt)
        elif tag == "title":
            txt = get_rich_text(child)
            if txt:
                buf.append(f"## {txt}")
        elif tag in ("fig", "fig-group"):
            cap = get_rich_text(child.find("./caption"))
            lbl = get_rich_text(child.find("./label"))
            buf.append(f"[FIGURE {lbl}: {cap}]")
        elif tag in ("sec", "app", "abstract"):
            buf.append(_walk(child, ctx))
    return "\n\n".join(buf)


def load_paper(nxml_path: Path) -> dict | None:
    """Load and parse a JATS NXML paper. Returns dict or None on parse failure."""
    try:
        tree = etree.parse(str(nxml_path))
        root = tree.getroot()
    except Exception:
        return None

    figures = extract_figures(root)
    tables = extract_tables(root)

    title_elem = root.find(".//front/article-meta/title-group/article-title")
    title = get_rich_text(title_elem)

    abstract_elem = root.find(".//front/article-meta/abstract")
    abstract_text = get_rich_text(abstract_elem)

    body_elem = root.find(".//body")

    sections: list[tuple[str, str]] = []
    if abstract_elem is not None:
        sections.append(("ABSTRACT", abstract_text))
    if body_elem is not None:
        ctx = {"stem": nxml_path.stem}
        for i, sec in enumerate(body_elem.findall("./sec"), start=1):
            stitle = get_rich_text(sec.find("./title")) or f"Section {i}"
            sec_text = _walk(sec, ctx)
            if sec_text.strip():
                sections.append((f"BODY: {stitle}", sec_text))

    body_preview = get_rich_text(body_elem)[:3000] if body_elem is not None else ""

    article_type = ""
    article_root = root.find(".//article")
    if article_root is not None:
        article_type = (article_root.get("article-type") or "").strip().lower()

    return {
        "nxml_path": str(nxml_path),
        "pmc_id": nxml_path.parent.name,
        "article_type": article_type,
        "title": title,
        "abstract": abstract_text,
        "body_preview": body_preview,
        "figures": figures,
        "tables": tables,
        "sections": sections,
    }


def find_nxml(pmc_dir: Path) -> Path | None:
    """Find the primary NXML file in a PMC extraction directory.

    Layout observed: {keyword}_extracted/{PMC_ID}/{PMC_ID}/<article>.nxml
    Also tolerates files sitting directly in pmc_dir.
    """
    if not pmc_dir.is_dir():
        return None
    # Try direct children first
    direct = sorted(pmc_dir.glob("*.nxml")) or sorted(pmc_dir.glob("*.xml"))
    for c in direct:
        if not c.name.startswith("."):
            return c
    # Try nested: pmc_dir/<any>/<article>.nxml
    for sub in pmc_dir.iterdir():
        if not sub.is_dir():
            continue
        nested = sorted(sub.glob("*.nxml")) or sorted(sub.glob("*.xml"))
        for c in nested:
            if not c.name.startswith("."):
                return c
    return None
