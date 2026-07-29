#!/usr/bin/env python3
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
"""Generate paired compression outputs for quality comparison."""

import asyncio
import json
import pathlib
import time

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from benchmarks.core import CACHE_FILE
from open_deep_research.prompts import (
    compress_research_system_prompt,
    compress_research_simple_human_message,
)
from open_deep_research.utils import get_today_str

OUT_FILE = pathlib.Path(__file__).parent / ".bench_compress_quality.json"
PAIRED_SEQS = [0, 3, 7, 12, 17]
MAX_TOKENS = 8192


def _build_messages(content: str, seq: int) -> list:
    topic = f"heart rate monitoring technology advances in {2025 - seq % 3}"
    tool_call = {
        "name": "tavily_search",
        "args": {"query": topic},
        "id": f"call_{seq}",
    }
    return [
        HumanMessage(content=f"Research topic: {topic}"),
        AIMessage(content="", tool_calls=[tool_call]),
        ToolMessage(content=content[:50000], name="tavily_search", tool_call_id=f"call_{seq}"),
        AIMessage(content=f"I found relevant information about {topic}. The key findings cover "
                  f"sensor technology, accuracy metrics, and clinical applications."),
    ]


async def gen_one(model_id: str, content: str, seq: int) -> dict:
    model = init_chat_model(model=model_id, max_tokens=MAX_TOKENS)
    researcher_msgs = _build_messages(content, seq)
    researcher_msgs.append(HumanMessage(content=compress_research_simple_human_message))
    sys_prompt = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=sys_prompt)] + researcher_msgs

    t0 = time.perf_counter()
    try:
        resp = await asyncio.wait_for(model.ainvoke(messages), timeout=90.0)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        ok = len(text) > 200
        raw = text if ok else f"TOO_SHORT({len(text)}): {text[:200]}"
    except asyncio.TimeoutError:
        ok, raw = False, "timeout"
    except Exception as e:
        ok, raw = False, f"{type(e).__name__}: {str(e)[:100]}"
    elapsed = time.perf_counter() - t0
    return {"elapsed_s": round(elapsed, 2), "raw": raw, "ok": ok}


async def main():
    cache = json.loads(CACHE_FILE.read_text())
    items = cache["items"]
    paired = []

    for seq in PAIRED_SEQS:
        item = items[seq % len(items)]
        print(f"seq={seq} ({len(item)} chars):")
        mimo_r = await gen_one("openai:mimo-v2.5", item, seq)
        ds_r = await gen_one("openai:deepseek-v4-flash", item, seq)
        paired.append({
            "seq": seq,
            "content_preview": item[:500],
            "content_len": len(item),
            "mimo_v2_5": mimo_r,
            "deepseek_v4_flash": ds_r,
        })
        print(f"  mimo:      {'OK' if mimo_r['ok'] else 'FAIL'} ({mimo_r['elapsed_s']:.1f}s)"
              + (f" {mimo_r['raw'][:80]}" if not mimo_r['ok'] else ""))
        print(f"  deepseek:  {'OK' if ds_r['ok'] else 'FAIL'} ({ds_r['elapsed_s']:.1f}s)"
              + (f" {ds_r['raw'][:80]}" if not ds_r['ok'] else ""))

    OUT_FILE.write_text(json.dumps(paired, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(paired)} pairs to {OUT_FILE}")


asyncio.run(main())
