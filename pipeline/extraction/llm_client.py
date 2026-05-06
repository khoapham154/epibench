"""HTTP client for the running vLLM OpenAI-compatible servers.

v4 changes:
- Pool of 8 vLLM servers on ports 8010-8017 (one Qwen3.5-35B-A3B instance per GPU).
- Round-robin dispatch per-request (not per-paper) so parallel STAGE_A calls
  inside one paper spread across servers.
- Global semaphore caps concurrent in-flight HTTP requests across all threads
  (default 128 = 8 servers * 16 slots; can go higher if KV cache allows).
"""
from __future__ import annotations
import itertools
import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable

import requests

# ----- Server pool configuration -----

def _parse_ports() -> list[int]:
    """VLLM_PORTS env: comma list ("8010,8011,...") or range ("8010:8017")."""
    raw = os.environ.get("VLLM_PORTS", "8010:8017")
    if ":" in raw:
        start, end = raw.split(":", 1)
        return list(range(int(start), int(end) + 1))
    return [int(p.strip()) for p in raw.split(",") if p.strip()]

VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")
VLLM_PORTS = _parse_ports()
VLLM_BASE_URLS = [f"http://{VLLM_HOST}:{p}/v1" for p in VLLM_PORTS]
VLLM_MODEL = os.environ.get("VLLM_MODEL", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")

DEFAULT_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", "300"))
MAX_RETRIES = int(os.environ.get("VLLM_MAX_RETRIES", "25"))
RETRY_BASE = float(os.environ.get("VLLM_RETRY_BASE", "1.0"))
RETRY_MAX_SLEEP = float(os.environ.get("VLLM_RETRY_MAX_SLEEP", "30"))
GLOBAL_CONCURRENCY = int(os.environ.get("VLLM_GLOBAL_CONCURRENCY", "128"))

RETRY_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

log = logging.getLogger("v3.llm")

# Thread-safe round-robin iterator over base URLs
_rr_lock = threading.Lock()
_rr_cycle = itertools.cycle(VLLM_BASE_URLS)

def _next_base_url() -> str:
    with _rr_lock:
        return next(_rr_cycle)

# Global semaphore caps concurrent in-flight HTTP requests across all threads.
_sem = threading.Semaphore(GLOBAL_CONCURRENCY)

# Per-server sessions for connection reuse
_sessions: dict[str, requests.Session] = {}
_sessions_lock = threading.Lock()

def _session_for(base_url: str) -> requests.Session:
    with _sessions_lock:
        s = _sessions.get(base_url)
        if s is None:
            s = requests.Session()
            _sessions[base_url] = s
        return s


def _sleep_backoff(attempt: int) -> float:
    """Exponential backoff with jitter, capped at RETRY_MAX_SLEEP."""
    base = min(RETRY_MAX_SLEEP, RETRY_BASE * (2 ** attempt))
    jitter = random.uniform(0, base * 0.3)
    return base + jitter


def chat_completion(
    messages: list[dict],
    *,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: float = DEFAULT_TIMEOUT,
    response_format_json: bool = True,
) -> str:
    """Blocking chat completion with retry + round-robin server + global concurrency cap."""
    payload: dict[str, Any] = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    last_err: str = ""
    for attempt in range(MAX_RETRIES):
        base_url = _next_base_url()
        session = _session_for(base_url)
        with _sem:
            try:
                r = session.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    timeout=timeout,
                )
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = f"{base_url} network: {e.__class__.__name__}: {e}"
                time.sleep(_sleep_backoff(attempt))
                continue

        if r.status_code == 200:
            try:
                data = r.json()
                return data["choices"][0]["message"]["content"] or ""
            except Exception as e:
                last_err = f"{base_url} parse: {e}"
                time.sleep(_sleep_backoff(attempt))
                continue

        if r.status_code in RETRY_STATUS_CODES:
            last_err = f"{base_url} http {r.status_code} {r.reason}"
            time.sleep(_sleep_backoff(attempt))
            continue

        # Non-retryable 4xx
        raise requests.HTTPError(f"{base_url} http {r.status_code} {r.reason}: {r.text[:200]}")

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {last_err}")


def parse_json(text: str) -> dict:
    """Extract JSON object from LLM response (tolerant of fences/prose)."""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}


def chat_completion_json(
    messages: list[dict],
    *,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> dict:
    raw = chat_completion(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format_json=True,
    )
    return parse_json(raw)


def parallel_map(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    max_workers: int = 64,
    desc: str | None = None,
    print_every: int = 1,
) -> list[Any]:
    items_list = list(items)
    if not items_list:
        return []
    results: list[Any] = [None] * len(items_list)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, item): i for i, item in enumerate(items_list)}
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                log.warning("parallel_map item %d failed: %s", idx, e)
                results[idx] = None
            done += 1
            if desc and (done % print_every == 0 or done == len(items_list)):
                print(f"   [{desc}] {done}/{len(items_list)}", flush=True)
    return results


def healthcheck() -> list[tuple[str, bool, str]]:
    """Return [(base_url, ok, message)] for all configured servers."""
    out: list[tuple[str, bool, str]] = []
    for url in VLLM_BASE_URLS:
        try:
            r = requests.get(f"{url}/models", timeout=5)
            if r.status_code == 200:
                out.append((url, True, "ok"))
            else:
                out.append((url, False, f"http {r.status_code}"))
        except Exception as e:
            out.append((url, False, str(e)[:60]))
    return out


if __name__ == "__main__":
    # Quick CLI: `python llm_client.py` prints server health
    for url, ok, msg in healthcheck():
        status = "OK" if ok else "FAIL"
        print(f"{status:4s} {url}  {msg}")
