"""Shared benchmark helpers: content fetch, JSON parse, constants."""

import asyncio
import json
import pathlib
import re

from open_deep_research.utils import tavily_search_async

N_CONTENT = 10
MAX_TOKENS = 8192
CACHE_FILE = pathlib.Path(__file__).parent / ".bench_cache.json"

_QUERIES = [
    "photoplethysmography PPG heart rate sensor technology 2025",
    "wearable biosensor optical heart rate monitoring clinical accuracy",
    "PPG signal processing motion artifact removal deep learning",
    "reflectance vs transmittance PPG sensor design comparison",
    "PPG heart rate variability HRV validation studies",
]

_JSON_RE = re.compile(r'\{.*"summary".*?"key_excerpts".*?\}', re.DOTALL)


async def fetch_content(n: int = N_CONTENT) -> list[str]:
    """Fetch real Tavily content (cached)."""
    if CACHE_FILE.exists():
        cached = json.loads(CACHE_FILE.read_text())
        items = cached.get("items", [])
        if len(items) >= n:
            print(f"  Using {len(items)} cached items from {CACHE_FILE}")
            return items[:n]

    results: list[str] = []
    round_num = 0
    config = {"configurable": {}}

    while len(results) < n:
        q = _QUERIES[round_num % len(_QUERIES)]
        round_num += 1
        print(f"  Tavily search #{round_num}: {q[:60]}...")
        try:
            search_out = await tavily_search_async([q], max_results=5, config=config)
            for batch in search_out:
                for r in batch.get("results", []):
                    rc = r.get("raw_content")
                    if rc and len(rc) > 200:
                        results.append(rc[:50000])
                        print(f"    got {len(rc)} chars ({len(results)}/{n})")
                        if len(results) >= n:
                            break
                if len(results) >= n:
                    break
        except Exception as e:
            print(f"    search failed: {e}")
            continue

    items = results[:n]
    CACHE_FILE.write_text(json.dumps({"items": items, "count": len(items)}))
    print(f"  Cached {len(items)} items to {CACHE_FILE}")
    return items


def parse_json_from_text(text: str) -> dict | None:
    """Extract JSON from model response text (structured output fallback)."""
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    text_stripped = text.strip()
    if text_stripped.startswith("{") and text_stripped.endswith("}"):
        try:
            return json.loads(text_stripped)
        except json.JSONDecodeError:
            pass
    text_clean = re.sub(r'^```(?:json)?\s*', '', text_stripped, flags=re.MULTILINE)
    text_clean = re.sub(r'\s*```$', '', text_clean, flags=re.MULTILINE)
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        return None
