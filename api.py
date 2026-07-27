"""Bot-agnostic research API. Wraps deep_researcher with pause/resume events."""

import asyncio
import re
import secrets
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from open_deep_research.deep_researcher import deep_researcher_builder
from open_deep_research.state import AgentState

# Model matrix (ref: odr-fork/run_once.py)
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

REPORTS_DIR = Path("reports")


# --- Event types (v1 contract) ---

@dataclass
class ClarifyQuestion:
    question: str


@dataclass
class BriefReady:
    brief: str


@dataclass
class SupervisorTick:
    research_iterations: int
    conduct_research_topics: list[str]


@dataclass
class SupervisorToolsDone:
    researchers_spawned: int
    new_urls: int


@dataclass
class ReportStarted:
    pass


@dataclass
class Done:
    report: str
    brief: str
    sources: list[str]
    report_path: str


@dataclass
class Failed:
    error: str


Event = ClarifyQuestion | BriefReady | SupervisorTick | SupervisorToolsDone | ReportStarted | Done | Failed


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def _extract_urls_from_messages(messages) -> list[str]:
    """Extract unique URLs from ToolMessage content via regex."""
    urls: set[str] = set()
    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            for match in re.findall(r"URL:\s*(\S+)", msg.content):
                urls.add(match)
    return list(urls)


async def research_stream(
    prompt: str,
    inbox: asyncio.Queue | None = None,
) -> AsyncIterator[Event]:
    """Stream research events. inbox=None = headless (auto-approve).

    Pause discrimination (after astream exhausts, via graph.get_state):
      - snap.next == () and no final_report and last msg is AIMessage → ClarifyQuestion
      - snap.next == ("research_supervisor",) and research_brief present → BriefReady
      - final_report present → Done
      - else → Failed
    """
    interactive = inbox is not None
    graph = deep_researcher_builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["research_supervisor"] if interactive else [],
    )

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            **MATRIX,
            "max_structured_output_retries": 3,
            "allow_clarification": True,
            "max_concurrent_research_units": 3,
            "search_api": "tavily",
            "max_researcher_iterations": 3,
            "max_react_tool_calls": 6,
        }
    }

    input_state = {"messages": [HumanMessage(content=prompt)]}

    while True:
        try:
            # Stream updates — extract supervisor progress events along the way
            # subgraphs=True yields 2-tuples: (namespace, {node_name: update})
            async for chunk in graph.astream(
                input_state, config, stream_mode="updates", subgraphs=True
            ):
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    _namespace, data = chunk
                elif isinstance(chunk, dict):
                    data = chunk
                else:
                    continue
                if not isinstance(data, dict):
                    continue

                for node_name, update in data.items():
                    if not isinstance(update, dict):
                        continue

                    if node_name == "supervisor":
                        iters = update.get("research_iterations", 0)
                        topics = []
                        for msg in update.get("supervisor_messages", []):
                            if hasattr(msg, "tool_calls"):
                                for tc in msg.tool_calls:
                                    if tc.get("name") == "ConductResearch":
                                        topics.append(tc["args"].get("research_topic", ""))
                        if iters:
                            yield SupervisorTick(
                                research_iterations=iters,
                                conduct_research_topics=topics,
                            )

                    elif node_name == "supervisor_tools":
                        tool_msgs = update.get("supervisor_messages", [])
                        spawned = sum(
                            1 for m in tool_msgs
                            if hasattr(m, "name") and m.name == "ConductResearch"
                        )
                        urls = len(_extract_urls_from_messages(tool_msgs))
                        if spawned or urls:
                            yield SupervisorToolsDone(
                                researchers_spawned=spawned,
                                new_urls=urls,
                            )

                    elif node_name == "final_report_generation":
                        if update.get("final_report"):
                            yield ReportStarted()

            # Check state
            snap = graph.get_state(config)
            state = snap.values
            messages = state.get("messages", [])
            last_msg = messages[-1] if messages else None
            final_report = state.get("final_report")
            research_brief = state.get("research_brief")
            next_nodes = snap.next

        except Exception as e:
            yield Failed(error=str(e))
            return

        # Case 1: Done — final_report present
        if final_report:
            sources = _extract_urls_from_messages(
                state.get("supervisor_messages", []) + messages
            )
            report_id = secrets.token_urlsafe(6)
            slug = _slugify(research_brief or prompt[:40])
            report_path = str(REPORTS_DIR / f"{slug}-{report_id}.md")
            REPORTS_DIR.mkdir(exist_ok=True)
            Path(report_path).write_text(final_report)

            yield Done(
                report=final_report,
                brief=research_brief or "",
                sources=sources,
                report_path=report_path,
            )
            return

        # Case 2: ClarifyQuestion — graph ended, no final_report, AIMessage pending
        if next_nodes == () and not final_report and isinstance(last_msg, AIMessage):
            if not interactive:
                # Headless: auto-answer with empty string to skip clarification
                input_state = {"messages": [HumanMessage(content="")]}
                continue

            yield ClarifyQuestion(question=last_msg.content)
            # Wait for user answer
            answer = await inbox.get()
            input_state = {"messages": [HumanMessage(content=answer)]}
            continue

        # Case 3: BriefReady — interrupt before research_supervisor
        if next_nodes == ("research_supervisor",) and research_brief:
            if not interactive:
                # Headless: auto-confirm
                input_state = None
                continue

            yield BriefReady(brief=research_brief)
            # Wait for confirm/cancel signal from bot (bot puts into inbox)
            signal = await inbox.get()
            if signal == "cancel":
                yield Failed(error="Cancelled by user")
                return
            input_state = None  # resume with None
            continue

        # Case 4: Indeterminate → Failed
        yield Failed(
            error=f"Indeterminate state: next={next_nodes}, "
            f"has_report={bool(final_report)}, last_msg_type={type(last_msg).__name__}"
        )
        return
