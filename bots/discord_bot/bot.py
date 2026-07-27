#!/usr/bin/env python3
"""Discord research bot — /research slash command, thread-per-run."""

import asyncio
import io
import logging
import os
import sys
import time
from pathlib import Path

import discord
import httpx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api import (
    research_stream,
    summarize_report,
    ClarifyQuestion,
    BriefReady,
    SupervisorTick,
    SupervisorToolsDone,
    ReportStarted,
    Done,
    Failed,
)

from rentry import publish_to_rentry

import re

load_dotenv()
TOKEN = os.environ["DISCORD_BOT_TOKEN"]

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("research-bot")
logger.setLevel(logging.INFO)

bot = discord.Bot()

# Per-user gate: user_id -> (task, thread)
_active: dict[int, tuple[asyncio.Task, discord.Thread]] = {}

_EMBED_COLOR = discord.Color(0x242429)
_RESEARCHING_GIF = "https://static2.klipy.com/ii/2711dd8a75a85be822d136ec94899b3f/97/01/z0RAnl2n.gif"

_groq = ChatGroq(model="llama-3.1-8b-instant", temperature=0) if os.environ.get("GROQ_API_KEY") else None
_topic_summaries: dict[str, str] = {}

async def _summarize_topic(topic: str) -> str:
    if topic in _topic_summaries:
        return _topic_summaries[topic]
    if _groq is None:
        return _truncate(topic, 100)
    try:
        resp = await _groq.ainvoke(
            [HumanMessage(content=(
              "<INSTRUCTIONS>"  
              "Rewrite this research topic as a one-line, verbose yet readable summary."
                "Summary only - do not answer."
              "First word must be an action verb or a direct noun."
              "10-20 words (STRICT)."
              "</INSTRUCTIONS>\n"
                f"<TOPIC>\n{topic}\n</TOPIC>"
            ))]
        )
        content = resp.content
        if not isinstance(content, str):
            content = str(content)
        summary = content.strip()
        if not summary:
            return _truncate(topic, 100)
        _topic_summaries[topic] = summary
        return summary
    except Exception:
        logger.warning("Groq topic summarize failed", exc_info=True)
        return _truncate(topic, 100)


def _extract_title(report: str, fallback: str) -> str:
    """Extract first # Title from markdown report. Fallback if missing."""
    m = re.search(r"^#\s+(.+)$", report, re.MULTILINE)
    if m:
        return m.group(1).strip()[:60]
    return fallback


def _build_status_embed(state: dict, gif: bool = True) -> discord.Embed:
    embed = discord.Embed(title="Researching", color=_EMBED_COLOR)
    if gif:
        embed.set_image(url=_RESEARCHING_GIF)
    if state["sources"]:
        embed.description = f"Sources: **` {state['sources']} `**"
    for i, topics in enumerate(state["phases"], 1):
        summarized = []
        for t in topics:
            if t in _topic_summaries:
                summarized.append(_topic_summaries[t])
            else:
                summarized.append(_truncate(t, 100))
        bullets = "\n\n".join(f"- {s}" for s in summarized)
        if len(bullets) > 1024:
            bullets = bullets[:1021] + "..."
        embed.add_field(name=f"❭ Research Phase {i}", value=bullets, inline=False)
    if state.get("phase") == "assessing":
        embed.add_field(name="❭ Analyzing Results", value="\u200b", inline=False)
    return embed


def _truncate(text: str, max_len: int) -> str:
    """Truncate at word boundary with ellipsis. No-op if ≤ max_len."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    space = truncated.rfind(" ")
    if space > 0:
        truncated = truncated[:space]
    return truncated.rstrip() + "..."


async def _run(bot, thread: discord.Thread, uid: int, prompt: str):
    inbox = asyncio.Queue()
    cancelled = asyncio.Event()

    state = {
        "sources": 0,
        "phases": [],  # list of topic lists per phase
        "phase": "starting",
    }
    live = {"sources": 0}
    last_tools_done_at = None

    # --- Cancel button (reusable) ---
    cancel_view = discord.ui.View(timeout=None)

    async def _on_cancel(interaction: discord.Interaction):
        cancelled.set()
        await inbox.put("cancel")
        try:
            await interaction.response.edit_message(content="Cancelled", view=None)
        except Exception:
            pass

    cb = discord.ui.Button(label="Cancel", style=discord.ButtonStyle.danger)
    cb.callback = _on_cancel
    cancel_view.add_item(cb)

    # Status message + updater — created when user confirms plan
    status_msg = None
    report_msg = None
    start_time = None
    upd_task = None

    # Placeholder message in thread — edited into research question embed
    brief_msg = await thread.send("**` Generating research question... `**")

    async def _start_updater():
        nonlocal upd_task
        _last_sig = None

        async def _upd():
            nonlocal _last_sig
            while not cancelled.is_set():
                state["sources"] = live.get("sources") or state["sources"]
                sig = (state["sources"], len(state["phases"]), state["phase"])
                if sig != _last_sig:
                    _last_sig = sig
                    try:
                        await status_msg.edit(embed=_build_status_embed(state), view=cancel_view)
                    except Exception:
                        pass
                try:
                    await asyncio.wait_for(cancelled.wait(), 1.5)
                except asyncio.TimeoutError:
                    pass

        upd_task = asyncio.create_task(_upd())


    try:
        async for event in research_stream(prompt, inbox, live=live):
            if cancelled.is_set():
                break

            # ----- ClarifyQuestion -----
            if isinstance(event, ClarifyQuestion):
                await thread.send(event.question)

                def _check(m):
                    return m.channel.id == thread.id and m.author.id == uid

                wt = asyncio.create_task(
                    bot.wait_for("message", check=_check, timeout=600)
                )
                ct = asyncio.create_task(cancelled.wait())
                done, pending = await asyncio.wait(
                    [wt, ct], return_when=asyncio.FIRST_COMPLETED
                )
                for t in pending:
                    t.cancel()

                if ct in done:
                    await inbox.put("cancel")
                    break

                try:
                    reply = wt.result()
                    await inbox.put(reply.content)
                except asyncio.TimeoutError:
                    await inbox.put("cancel")
                    await thread.send("Timed out — cancelled.")
                    break

            # ----- BriefReady -----
            elif isinstance(event, BriefReady):
                bv = discord.ui.View(timeout=600)

                async def _start(interaction: discord.Interaction):
                    nonlocal status_msg, start_time
                    bv.stop()  # cancel timeout timer so _bv_timeout doesn't fire later
                    await inbox.put("start")
                    await interaction.response.edit_message(view=None)
                    start_time = time.monotonic()
                    status_msg = await thread.send(embed=_build_status_embed(state), view=cancel_view)
                    await _start_updater()

                async def _cancel_brief(interaction: discord.Interaction):
                    cancelled.set()
                    await inbox.put("cancel")
                    try:
                        await interaction.response.edit_message(view=None)
                    except Exception:
                        pass

                async def _bv_timeout():
                    cancelled.set()
                    await inbox.put("cancel")
                    try:
                        await bv.message.edit(content="Timed out — cancelled.", view=None)
                    except Exception:
                        pass

                bv.on_timeout = _bv_timeout

                sb = discord.ui.Button(label="Start", style=discord.ButtonStyle.primary)
                sb.callback = _start
                cbb = discord.ui.Button(
                    label="Cancel", style=discord.ButtonStyle.danger
                )
                cbb.callback = _cancel_brief
                bv.add_item(sb)
                bv.add_item(cbb)

                brief_embed = discord.Embed(title="Research Question", description=event.brief, color=_EMBED_COLOR)
                try:
                    await brief_msg.edit(content=None, embed=brief_embed, view=bv)
                except Exception:
                    await thread.send(embed=brief_embed, view=bv)

            # ----- SupervisorTick -----
            elif isinstance(event, SupervisorTick):
                if event.conduct_research_topics:
                    state["phase"] = "researching"
                    state["phases"].append(event.conduct_research_topics)
                    for t in event.conduct_research_topics:
                        await _summarize_topic(t)
                    if status_msg:
                        await status_msg.edit(embed=_build_status_embed(state), view=cancel_view)

            # ----- SupervisorToolsDone -----
            elif isinstance(event, SupervisorToolsDone):
                state["sources"] = event.sources
                state["phase"] = "assessing"
                last_tools_done_at = time.monotonic()
                if status_msg:
                    await status_msg.edit(embed=_build_status_embed(state), view=cancel_view)

            # ----- ReportStarted -----
            elif isinstance(event, ReportStarted):
                state["phase"] = "writing"
                if upd_task:
                    upd_task.cancel()
                report_msg = await thread.send("https://klipy.com/gifs/fire-writing")
                # Research is done — update status embed with elapsed time, remove gif
                if status_msg and start_time:
                    elapsed = int(time.monotonic() - start_time)
                    if elapsed >= 3600:
                        footer = f"Research time: {elapsed // 3600}h {(elapsed % 3600) // 60}m"
                    elif elapsed >= 60:
                        footer = f"Research time: {elapsed // 60}m {elapsed % 60}s"
                    else:
                        footer = f"Research time: {elapsed}s"
                    try:
                        embed = _build_status_embed(state, gif=False)
                        embed.set_footer(text=footer)
                        await status_msg.edit(embed=embed, view=None)
                    except Exception:
                        pass

            # ----- Done -----
            elif isinstance(event, Done):
                cancelled.set()

                # Final source count from regex on all messages (accurate)
                state["sources"] = len(event.sources)
                if status_msg:
                    try:
                        await status_msg.edit(embed=_build_status_embed(state, gif=False))
                    except Exception:
                        pass

                # Publish to rentry (best-effort), fallback to file upload
                brief = event.brief
                fallback = _truncate(brief, 100)
                title = _extract_title(event.report, fallback)
                rentry_url = publish_to_rentry(event.report, title)
                if rentry_url:
                    kwargs = {"content": f"📄 **Report:** {rentry_url}"}
                else:
                    kwargs = {
                        "content": "📄 Report uploaded as file.",
                        "file": discord.File(io.BytesIO(event.report.encode()), filename="research-report.md"),
                    }

                # TL;DR button
                tldr_view = discord.ui.View(timeout=3600)
                tldr_btn = discord.ui.Button(label="TL;DR", style=discord.ButtonStyle.secondary)

                async def _tldr(interaction: discord.Interaction):
                    await interaction.response.defer(ephemeral=False)
                    tldr_btn.disabled = True
                    try:
                        await interaction.message.edit(view=tldr_view)
                    except Exception:
                        pass
                    placeholder = await interaction.channel.send("**` Summarizing... `**")
                    try:
                        summary = await summarize_report(event.report, event.brief)
                        if len(summary) > 4096:
                            summary = summary[:4096].rsplit(" ", 1)[0] + "…"
                        embed = discord.Embed(title="TL;DR", description=summary, color=_EMBED_COLOR)
                        await placeholder.edit(content=None, embed=embed)
                    except Exception:
                        logger.warning("TL;DR failed", exc_info=True)
                        await placeholder.edit(content="TL;DR failed. Try again later.")
                        tldr_btn.disabled = False
                        try:
                            await interaction.message.edit(view=tldr_view)
                        except Exception:
                            pass

                tldr_btn.callback = _tldr
                tldr_view.add_item(tldr_btn)
                kwargs["view"] = tldr_view

                if report_msg:
                    try:
                        await report_msg.edit(**kwargs)
                    except Exception:
                        await thread.send(**kwargs)
                else:
                    await thread.send(**kwargs)
                return

            # ----- Failed -----
            elif isinstance(event, Failed):
                cancelled.set()
                if upd_task:
                    upd_task.cancel()
                logger.error("Research failed:\n%s", event.error)
                if status_msg:
                    await status_msg.edit(embed=discord.Embed(title="Failed", description=event.error[:2048], color=_EMBED_COLOR), view=None)
                else:
                    await thread.send(event.error)
                return

    except asyncio.CancelledError:
        cancelled.set()
        if upd_task:
            upd_task.cancel()
        if status_msg:
            try:
                await status_msg.edit(embed=discord.Embed(title="Cancelled", color=_EMBED_COLOR), view=None)
            except Exception:
                pass
        raise
    finally:
        cancelled.set()
        if upd_task and not upd_task.done():
            upd_task.cancel()


@bot.slash_command(description="Run deep research on a topic. Results in a thread.")
async def research(
    ctx: discord.ApplicationContext,
    prompt: str,
):
    """Run deep research on a topic. Results delivered in a thread."""
    uid = ctx.author.id

    # Per-user gate
    if uid in _active:
        t, th = _active[uid]
        if not t.done():
            await ctx.respond(
                f"You already have research running in {th.mention}",
                ephemeral=True,
            )
            return

    if not isinstance(ctx.channel, discord.TextChannel):
        await ctx.respond("Must be used in a text channel.", ephemeral=True)
        return

    thread_name = f"🔬 {prompt[:40]}"
    prompt_display = _truncate(prompt, 200)
    await ctx.respond(f"🔬 Researching: {prompt_display}")
    start_msg = await ctx.interaction.original_response()
    thread = await start_msg.create_thread(name=thread_name)

    task = asyncio.create_task(_run(bot, thread, uid, prompt))
    _active[uid] = (task, thread)

    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        if uid in _active and _active[uid][0] is task:
            del _active[uid]


@bot.slash_command(description="Show Tavily API usage for current billing cycle.")
async def usage(ctx: discord.ApplicationContext):
    """Fetch and display Tavily API usage."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        await ctx.respond("TAVILY_API_KEY not set.", ephemeral=True)
        return

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.tavily.com/usage",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code != 200:
                await ctx.respond(f"Tavily API error: {resp.status_code}", ephemeral=True)
                return
            data = resp.json()
    except Exception as e:
        await ctx.respond(f"Failed to fetch usage: {e}", ephemeral=True)
        return

    account = data.get("account", {})

    embed = discord.Embed(
        title="Tavily API Usage",
        color=_EMBED_COLOR,
    )

    plan = account.get("current_plan", "—")
    pu = account.get("plan_usage", 0)
    pl = account.get("plan_limit")
    if pl:
        embed.add_field(name=f"Plan ({plan})", value=f"{pu} / {pl} credits used", inline=False)

    await ctx.respond(embed=embed, ephemeral=True)


@bot.event
async def on_ready():
    logger.info("Logged in as %s", bot.user)


if __name__ == "__main__":
    bot.run(TOKEN)
