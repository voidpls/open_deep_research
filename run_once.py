"""Run stock open_deep_research once with a prompt file.

Usage:
    python run_once.py benchmark/prompts/my_prompt.txt

Reads .env for API keys. Outputs report to stdout and saves to reports/<slug>.md.
"""

import argparse
import asyncio
import os
import pathlib
import uuid
import re
from dotenv import load_dotenv

load_dotenv(override=True)  # override shell env (e.g. Claude Code ANTHROPIC_BASE_URL)

from open_deep_research.deep_researcher import deep_researcher_builder
from langgraph.checkpoint.memory import MemorySaver


# Testing matrix — swap values here when changing model assignment.
# Keys match Configuration field names 1:1.
MATRIX = {
    "structured_model": "openai:mimo-v2.5-pro",
    "structured_model_max_tokens": 10000,
    "research_model": "openai:minimax-m3",
    "research_model_max_tokens": 10000,
    "summarization_model": "openai:mimo-v2.5",
    "summarization_model_max_tokens": 8192,
    "compression_model": "openai:mimo-v2.5",
    "compression_model_max_tokens": 10000,
    "final_report_model": "openai:minimax-m3",
    "final_report_model_max_tokens": 10000,
}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


async def run(prompt: str, name: str = "report"):
    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            **MATRIX,
            "max_structured_output_retries": 3,
            "allow_clarification": False,
            "max_concurrent_research_units": 3,
            "search_api": "tavily",
            "max_researcher_iterations": 4,
            "max_react_tool_calls": 6,
            "max_content_length": 10000,
        }
    }

    print(f"[run_once] Starting research for: {prompt[:120]}...")
    print(f"[run_once] Matrix: { {k: v for k, v in MATRIX.items() if k.endswith('_model')} }")
    print(f"[run_once] Base URL: {os.getenv('OPENAI_BASE_URL')}")
    print()

    final_state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config,
    )

    report = final_state.get("final_report", "(no final_report in final state)")
    # Also dump all keys for debugging
    print(f"\n[run_once] Final state keys: {list(final_state.keys())}")
    print(f"{'='*60}")
    print(f"REPORT: {name}")
    print(f"{'='*60}\n")
    print(report)

    # Save report
    out_dir = pathlib.Path("reports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{name}.md"
    out_path.write_text(report)
    print(f"\n[run_once] Saved to {out_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run stock open_deep_research once")
    parser.add_argument("prompt_file", help="Path to text file containing the prompt")
    args = parser.parse_args()

    prompt_path = pathlib.Path(args.prompt_file)
    if not prompt_path.exists():
        print(f"Error: prompt file not found: {prompt_path}")
        return 1

    prompt = prompt_path.read_text().strip()
    name = slugify(prompt_path.stem)
    asyncio.run(run(prompt, name))
    return 0


if __name__ == "__main__":
    exit(main())
