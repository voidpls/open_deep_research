"""Smoke test: run a trivial prompt through odr-stock, print per-node latencies."""
import asyncio, json, re, time, uuid
from dotenv import load_dotenv
# Run from odr-stock root; loads root .env
load_dotenv(override=True)
from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.deep_researcher import deep_researcher_builder

PROMPT = "What is the capital of Japan? One sentence only."

CAPS = {"max_researcher_iterations": 1, "max_concurrent_research_units": 1, "max_react_tool_calls": 2}

async def main():
    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4()), **CAPS,
              "allow_clarification": False, "search_api": "tavily"}}

    print(f"Prompt: {PROMPT}")
    print(f"Base URL: {__import__('os').getenv('OPENAI_BASE_URL', 'unset')}")
    print()

    t0 = time.perf_counter(); last = t0; node_times = {}
    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": PROMPT}]},
        config, stream_mode="updates",
    ):
        now = time.perf_counter()
        for node in chunk:
            dt = now - last
            node_times[node] = node_times.get(node, 0.0) + dt
            print(f"  [{node}] dt={dt:.1f}s")
        last = now

    elapsed = time.perf_counter() - t0
    values = graph.get_state(config).values
    report = values.get("final_report") or ""

    print()
    print(json.dumps({
        "elapsed_s": round(elapsed, 2),
        "node_times": {k: round(v, 3) for k, v in node_times.items()},
        "report_len": len(report),
        "think_tag_count": len(re.findall(r"<think>", report)),
        "report_preview": report[:300],
    }, indent=2))

asyncio.run(main())
