#!/usr/bin/env python3
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
"""Generate paired summaries for quality comparison (text+parse path)."""

import asyncio, json, pathlib, time
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from open_deep_research.utils import summarize_webpage_prompt, get_today_str

from benchmarks.core import CACHE_FILE, parse_json_from_text

OUT_FILE = pathlib.Path(__file__).parent / ".bench_quality.json"
PAIRED_SEQS = [0, 3, 12, 13, 16]

JSON_INSTRUCTION = (
    "\n\nYou MUST return ONLY valid JSON with fields: summary (string), key_excerpts (string). "
    "No markdown fences, no prose, no reasoning — just the JSON object."
)


async def gen_one(model_id: str, item: str) -> dict:
    model = init_chat_model(model=model_id, max_tokens=8192)
    prompt = summarize_webpage_prompt.format(webpage_content=item, date=get_today_str()) + JSON_INSTRUCTION
    t0 = time.perf_counter()
    try:
        resp = await asyncio.wait_for(model.ainvoke([HumanMessage(content=prompt)]), timeout=65.0)
        parsed = parse_json_from_text(resp.content)
        ok = parsed is not None
        raw = resp.content if ok else f"PARSE_FAILED: {resp.content[:200]}"
    except asyncio.TimeoutError:
        ok, parsed, raw = False, None, "timeout"
    except Exception as e:
        ok, parsed, raw = False, None, f"{type(e).__name__}: {str(e)[:100]}"
    elapsed = time.perf_counter() - t0
    return {"elapsed_s": round(elapsed, 2), "raw": raw, "parsed": parsed, "ok": ok}


async def main():
    cache = json.loads(CACHE_FILE.read_text())
    items = cache["items"]
    paired = []
    for seq in PAIRED_SEQS:
        item = items[seq % len(items)]
        print(f"seq={seq} ({len(item)} chars):")
        mimo_r = await gen_one("openai:mimo-v2.5", item)
        ds_r = await gen_one("openai:deepseek-v4-flash", item)
        paired.append({
            "seq": seq,
            "content_preview": item[:500],
            "content_len": len(item),
            "mimo_v2_5": mimo_r,
            "deepseek_v4_flash": ds_r,
        })
        print(f"  mimo:      {'OK' if mimo_r['ok'] else 'FAIL'} ({mimo_r['elapsed_s']:.1f}s)")
        print(f"  deepseek:  {'OK' if ds_r['ok'] else 'FAIL'} ({ds_r['elapsed_s']:.1f}s)")

    OUT_FILE.write_text(json.dumps(paired, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(paired)} pairs to {OUT_FILE}")


asyncio.run(main())
