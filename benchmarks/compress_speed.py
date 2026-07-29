#!/usr/bin/env python3
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
"""Benchmark compression throughput: plain text in, plain text out."""

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

from benchmarks.core import CACHE_FILE, fetch_content
from open_deep_research.prompts import (
    compress_research_system_prompt,
    compress_research_simple_human_message,
)
from open_deep_research.utils import get_today_str

N = 10  # calls per model
MAX_TOKENS = 8192
_RESULTS = pathlib.Path(__file__).parent / ".bench_compress_speed.json"

MODELS = {
    "mimo-v2.5": "openai:mimo-v2.5",
    "deepseek-v4-flash": "openai:deepseek-v4-flash",
}


def _build_messages(content: str, seq: int) -> list:
    """Build synthetic researcher conversation: topic -> tool call -> result -> analysis."""
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


async def bench_one(model_key: str, model_id: str, content: str, seq: int) -> dict:
    """Run one compression call."""
    t0 = time.perf_counter()
    ok = False
    error: str | None = None

    try:
        model = init_chat_model(model=model_id, max_tokens=MAX_TOKENS)

        researcher_msgs = _build_messages(content, seq)
        researcher_msgs.append(HumanMessage(content=compress_research_simple_human_message))
        sys_prompt = compress_research_system_prompt.format(date=get_today_str())
        messages = [SystemMessage(content=sys_prompt)] + researcher_msgs

        resp = await asyncio.wait_for(model.ainvoke(messages), timeout=90.0)
        text = resp.content
        if isinstance(text, str) and len(text) > 200:
            ok = True
        else:
            error = "empty_or_too_short"
    except asyncio.TimeoutError:
        error = "timeout"
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)[:120]}"

    elapsed = time.perf_counter() - t0
    return {
        "model": model_key,
        "seq": seq,
        "ok": ok,
        "error": error,
        "elapsed_s": round(elapsed, 3),
    }


async def run_benchmark(items: list[str]) -> list[dict]:
    """Run N calls per model, 2 at a time."""
    total = N * len(MODELS)
    results: list[dict] = []
    done = 0
    sem = asyncio.Semaphore(len(MODELS))

    async def worker(model_key: str, model_id: str, item: str, seq: int):
        async with sem:
            r = await bench_one(model_key, model_id, item, seq)
            nonlocal done
            done += 1
            results.append(r)
            ok_count = sum(1 for x in results if x.get("ok") and x["model"] == model_key)
            total_count = sum(1 for x in results if x["model"] == model_key)
            print(f"  [{done}/{total}] {model_key:20s} #{seq:2d}: {'OK' if r['ok'] else 'FAIL':4s} "
                  f"({r['elapsed_s']:.2f}s)  [{ok_count}/{total_count}]"
                  + (f" {r['error'][:60]}" if r['error'] else ""))

    tasks = []
    for seq in range(N):
        item = items[seq % len(items)]
        for model_key, model_id in MODELS.items():
            tasks.append(worker(model_key, model_id, item, seq))

    await asyncio.gather(*tasks)
    return results


def report(results: list[dict]):
    for model_key, model_id in MODELS.items():
        rows = [r for r in results if r["model"] == model_key]
        total = len(rows)
        ok = sum(1 for r in rows if r["ok"])
        fail = total - ok
        times = [r["elapsed_s"] for r in rows if r["ok"]]

        errors: dict[str, int] = {}
        for r in rows:
            if r["error"]:
                err_key = r["error"].split(":")[0] if ":" in r["error"] else r["error"]
                errors[err_key] = errors.get(err_key, 0) + 1

        print(f"\n── {model_key} ──")
        print(f"  {ok}/{total} ok  ({fail} failed)")
        if errors:
            for e, c in sorted(errors.items(), key=lambda kv: -kv[1]):
                print(f"    {e}: {c}")
        if times:
            avg = sum(times) / len(times)
            p50 = sorted(times)[len(times) // 2]
            print(f"  avg: {avg:.2f}s  p50: {p50:.2f}s  "
                  f"min: {min(times):.2f}s  max: {max(times):.2f}s")

    print(f"\n{'='*50}")
    for model_key in MODELS:
        rows = [r for r in results if r["model"] == model_key]
        ok = sum(1 for r in rows if r["ok"])
        total = len(rows)
        times = [r["elapsed_s"] for r in rows if r["ok"]]
        avg = sum(times) / len(times) if times else 0
        print(f"  {model_key:20s}  {ok:2d}/{total}  avg {avg:.2f}s")

    _RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nRaw: {_RESULTS}")


async def main():
    print("=== Compression Model Benchmark ===\n")
    for k, v in MODELS.items():
        print(f"  {k}: {v}")
    print(f"  Calls per model: {N}\n")

    print("Step 1: Fetching Tavily content...")
    items = await fetch_content()
    print(f"  Got {len(items)} items\n")

    print(f"Step 2: Running {N * len(MODELS)} compression calls (2-way parallel)...")
    results = await run_benchmark(items)

    print("\nStep 3: Results")
    report(results)


if __name__ == "__main__":
    asyncio.run(main())
