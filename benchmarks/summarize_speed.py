#!/usr/bin/env python3
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
"""Benchmark summarization throughput: structured vs text+parse paths."""

import asyncio
import json
import pathlib
import time

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from open_deep_research.state import Summary

from benchmarks.core import CACHE_FILE, MAX_TOKENS, N_CONTENT, fetch_content, parse_json_from_text

N_PER_MODEL = 20

MODELS = {
    "mimo-v2.5": {
        "id": "openai:mimo-v2.5",
        "structured": True,   # supports .with_structured_output
    },
    "deepseek-v4-flash": {
        "id": "openai:deepseek-v4-flash",
        "structured": False,  # Console Go rejects response_format
    },
}


async def bench_one(model_key: str, model_cfg: dict, item: str, seq: int) -> dict:
    from open_deep_research.utils import summarize_webpage_prompt, get_today_str
    t0 = time.perf_counter()
    ok = False
    error: str | None = None

    try:
        base_model = init_chat_model(
            model=model_cfg["id"],
            max_tokens=MAX_TOKENS,
        )

        if model_cfg["structured"]:
            model = base_model.with_structured_output(Summary).with_retry(stop_after_attempt=1)
            prompt = summarize_webpage_prompt.format(webpage_content=item, date=get_today_str())
            result = await asyncio.wait_for(
                model.ainvoke([HumanMessage(content=prompt)]),
                timeout=65.0,
            )
            formatted = (
                f"<summary>\n{result.summary}\n</summary>\n\n"
                f"<key_excerpts>\n{result.key_excerpts}\n</key_excerpts>"
            )
            ok = True
        else:
            prompt = (
                summarize_webpage_prompt.format(webpage_content=item, date=get_today_str())
                + "\n\nYou MUST return ONLY valid JSON with fields: summary (string), key_excerpts (string)."
                " No markdown fences, no prose, no reasoning — just the JSON object."
            )
            raw = await asyncio.wait_for(
                base_model.ainvoke([HumanMessage(content=prompt)]),
                timeout=65.0,
            )
            parsed = parse_json_from_text(raw.content)
            if parsed and isinstance(parsed.get("summary"), str) and isinstance(parsed.get("key_excerpts"), str):
                formatted = (
                    f"<summary>\n{parsed['summary']}\n</summary>\n\n"
                    f"<key_excerpts>\n{parsed['key_excerpts']}\n</key_excerpts>"
                )
                ok = True
            else:
                error = f"parse_failed: content_preview={raw.content[:100]!r}"
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
    """Run N_PER_MODEL calls per model, 2 at a time (one per model)."""
    total = N_PER_MODEL * len(MODELS)
    results: list[dict] = []
    done = 0

    sem = asyncio.Semaphore(len(MODELS))

    async def worker(model_key: str, model_cfg: dict, item: str, seq: int):
        async with sem:
            r = await bench_one(model_key, model_cfg, item, seq)
            nonlocal done
            done += 1
            results.append(r)
            count_ok = sum(1 for x in results if x.get("ok") and x["model"] == model_key)
            count_total = sum(1 for x in results if x["model"] == model_key)
            print(f"  [{done}/{total}] {model_key:20s} #{seq:2d}: {'OK' if r['ok'] else 'FAIL':4s} "
                  f"({r['elapsed_s']:.2f}s)  [{count_ok}/{count_total}]"
                  + (f" {r['error'][:60]}" if r['error'] else ""))

    tasks = []
    for seq in range(N_PER_MODEL):
        item = items[seq % len(items)]
        for model_key, model_cfg in MODELS.items():
            tasks.append(worker(model_key, model_cfg, item, seq))

    await asyncio.gather(*tasks)
    return results


def report(results: list[dict]):
    for model_key, model_cfg in MODELS.items():
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

        print(f"\n── {model_key} ({'structured' if model_cfg['structured'] else 'text+parse'}) ──")
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

    out = pathlib.Path(__file__).parent / ".bench_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nRaw: {out}")


async def main():
    print("=== Summarization Model Benchmark ===\n")
    for k, v in MODELS.items():
        print(f"  {k}: {v['id']} ({'structured' if v['structured'] else 'text+parse'})")
    print(f"  Content items: {N_CONTENT}, Calls per model: {N_PER_MODEL}\n")

    print("Step 1: Fetching real Tavily content...")
    items = await fetch_content(N_CONTENT)
    print(f"  Got {len(items)} items ({sum(len(i) for i in items)} total chars)\n")

    print(f"Step 2: Running {N_PER_MODEL * len(MODELS)} summarization calls (2-way parallel)...")
    results = await run_benchmark(items)

    print("\nStep 3: Results")
    report(results)


if __name__ == "__main__":
    asyncio.run(main())
