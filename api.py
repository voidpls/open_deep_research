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

from open_deep_research.deep_researcher import deep_researcher_builder, model_profile, _llm_config
from open_deep_research.state import AgentState
from open_deep_research.utils import get_api_key_for_model, strip_think_tags

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
    sources: int  # cumulative unique URLs so far


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


_URL_RE = re.compile(r"https?://[^\s<>\"']+")


def _extract_urls_from_text(text: str) -> set[str]:
    if not text:
        return set()
    return {m.rstrip(".,);]!>") for m in _URL_RE.findall(text)}


def _extract_urls_from_messages(messages) -> list[str]:
    urls: set[str] = set()
    for msg in messages:
        content = getattr(msg, "content", None)
        if content is None:
            continue
        urls |= _extract_urls_from_text(content if isinstance(content, str) else str(content))
    return list(urls)


def _extract_urls_from_raw_notes(raw_notes) -> set[str]:
    urls: set[str] = set()
    for note in raw_notes or []:
        urls |= _extract_urls_from_text(note if isinstance(note, str) else str(note))
    return urls


_TLDR_PROMPT = """You write a TL;DR of a research report for a Discord message.

Research brief (what was asked):
{brief}

Full report:
{report}

Rules:
- ~200 words (180-220). Stop when done; do not pad.
- Answer the brief: lead with the bottom line, then the key findings and caveats that matter.
- Stay grounded only in the report; do not invent facts or citations.
- Inline markdown only: **bold**, *italic*, and `-` bullets if useful. No headings (#), no code blocks, no tables, no horizontal rules, no raw URLs dump.
- No preamble ("Here is a summary") and no closing offer to expand.
"""


async def summarize_report(report: str, brief: str) -> str:
    """Call ODR final_report_model with TL;DR prompt. Returns ~200 word summary.

    Matches deep_researcher.final_report_generation pattern:
    model_profile.with_config(_llm_config(...)).ainvoke(...) + strip_think_tags.
    """
    model_id = MATRIX["final_report_model"]
    report_in = report if len(report) <= 120_000 else report[:120_000] + "\n\n[truncated]"
    cfg = _llm_config(
        model=model_id,
        max_tokens=8000,
        api_key=get_api_key_for_model(model_id, {}),
        tags=["langsmith:nostream"],
    )
    prompt = _TLDR_PROMPT.replace("{brief}", brief or "(none)").replace("{report}", report_in)
    try:
        resp = await model_profile.with_config(cfg).ainvoke([HumanMessage(content=prompt)])
    except Exception:
        import logging
        logging.getLogger("research-bot").warning("TL;DR model call failed", exc_info=True)
        return ""
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return strip_think_tags(text).strip()


async def research_stream(
    prompt: str,
    inbox: asyncio.Queue | None = None,
    live: dict | None = None,
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

    seen_urls: set[str] = set()
    input_state = {"messages": [HumanMessage(content=prompt)]}

    import open_deep_research.utils as _utils

    _orig_async = _utils.tavily_search_async

    async def _count_async(*args, **kwargs):
        search_results = await _orig_async(*args, **kwargs)
        found = set()
        for response in search_results or []:
            for result in response.get("results") or []:
                if url := result.get("url"):
                    found.add(url)
        if found:
            seen_urls.update(found)
            if live is not None:
                live["sources"] = len(seen_urls)
        return search_results

    _utils.tavily_search_async = _count_async
    try:
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
                            # supervisor_messages missing = supervisor ended (ResearchComplete / no_tool_calls / max_iterations)
                            if "supervisor_messages" not in update:
                                yield ReportStarted()
                                continue
                            tool_msgs = update.get("supervisor_messages", [])
                            spawned = sum(
                                1 for m in tool_msgs
                                if hasattr(m, "name") and m.name == "ConductResearch"
                            )
                            if spawned:
                                yield SupervisorToolsDone(
                                    researchers_spawned=spawned,
                                    new_urls=0,
                                    sources=len(seen_urls),
                                )

                        elif node_name == "final_report_generation":
                            pass  # ReportStarted already yielded when supervisor ended

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
                sources = list(
                    set(
                        _extract_urls_from_messages(
                            state.get("supervisor_messages", []) + messages
                        )
                    )
                    | _extract_urls_from_raw_notes(state.get("raw_notes", []))
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
    finally:
        _utils.tavily_search_async = _orig_async
