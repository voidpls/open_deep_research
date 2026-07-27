#!/usr/bin/env python3
"""Discord research bot — /research slash command, thread-per-run."""

import asyncio
import logging
import os
import sys
from pathlib import Path

import discord
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api import (
    research_stream,
    ClarifyQuestion,
    BriefReady,
    SupervisorTick,
    SupervisorToolsDone,
    ReportStarted,
    Done,
    Failed,
)

from rentry import publish_to_rentry

load_dotenv()
TOKEN = os.environ["DISCORD_BOT_TOKEN"]

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("research-bot")
logger.setLevel(logging.INFO)

bot = discord.Bot()

# Per-user gate: user_id -> (task, thread)
_active: dict[int, tuple[asyncio.Task, discord.Thread]] = {}

_EMBED_COLOR = discord.Color(0x313338)

_groq = ChatGroq(model="llama-3.1-8b-instant", temperature=0) if os.environ.get("GROQ_API_KEY") else None
_topic_summaries: dict[str, str] = {}

async def _summarize_topic(topic: str) -> str:
    if topic in _topic_summaries:
        return _topic_summaries[topic]
    if _groq is None:
        return topic[:100]
    try:
        resp = await _groq.ainvoke(
            [HumanMessage(content=(
                "Rewrite this research topic as a short status label (≤15 words). "
                "Name the subject only — do not answer, refuse, or add facts.\n"
                f"Topic: {topic}"
            ))]
        )
        content = resp.content
        if not isinstance(content, str):
            content = str(content)
        summary = content.strip()[:100]
        if not summary:
            return topic[:100]
        _topic_summaries[topic] = summary
        return summary
    except Exception:
        logger.warning("Groq topic summarize failed", exc_info=True)
        return topic[:100]


def _build_status_embed(state: dict) -> discord.Embed:
    embed = discord.Embed(title="Researching", color=_EMBED_COLOR)
    for i, topics in enumerate(state["phases"], 1):
        summarized = []
        for t in topics:
            if t in _topic_summaries:
                summarized.append(_topic_summaries[t])
            else:
                summarized.append(t[:100])
        bullets = "\n\n".join(f"◘ **{s}**" for s in summarized)
        if len(bullets) > 1024:
            bullets = bullets[:1021] + "..."
        embed.add_field(name=f"Phase {i}", value=bullets, inline=False)
    footer_parts = []
    if state["researchers"]:
        footer_parts.append(f"{state['researchers']} researchers")
    if state["sources"]:
        footer_parts.append(f"{state['sources']} sources")
    if footer_parts:
        embed.set_footer(text=" · ".join(footer_parts))
    return embed


async def _run(bot, thread: discord.Thread, uid: int, prompt: str):
    inbox = asyncio.Queue()
    cancelled = asyncio.Event()

    state = {
        "researchers": 0,
        "sources": 0,
        "phases": [],  # list of topic lists per phase
    }

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
    upd_task = None

    async def _start_updater():
        nonlocal upd_task

        async def _upd():
            while not cancelled.is_set():
                try:
                    await status_msg.edit(embed=_build_status_embed(state), view=cancel_view)
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(cancelled.wait(), 5)
                except asyncio.TimeoutError:
                    pass

        upd_task = asyncio.create_task(_upd())


    try:
        async for event in research_stream(prompt, inbox):
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
                    nonlocal status_msg
                    await inbox.put("start")
                    await interaction.response.edit_message(view=None)
                    # Show empty "Researching" embed immediately
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

                await thread.send(event.brief, view=bv)

            # ----- SupervisorTick -----
            elif isinstance(event, SupervisorTick):
                if event.conduct_research_topics:
                    state["phases"].append(event.conduct_research_topics)
                    for t in event.conduct_research_topics:
                        logger.info("topic: %s", t)
                        await _summarize_topic(t)
                    if status_msg:
                        await status_msg.edit(embed=_build_status_embed(state), view=cancel_view)

            # ----- SupervisorToolsDone -----
            elif isinstance(event, SupervisorToolsDone):
                state["researchers"] = event.researchers_spawned
                state["sources"] = event.new_urls
                if status_msg:
                    await status_msg.edit(embed=_build_status_embed(state), view=cancel_view)

            # ----- ReportStarted -----
            elif isinstance(event, ReportStarted):
                if upd_task:
                    upd_task.cancel()
                report_msg = await thread.send("https://klipy.com/gifs/fire-writing")

            # ----- Done -----
            elif isinstance(event, Done):
                cancelled.set()

                # Publish to rentry (best-effort)
                title = await _summarize_topic(event.brief) or event.brief[:60]
                rentry_url = publish_to_rentry(event.report, title)
                content = f"📄 **Report:** {rentry_url}" if rentry_url else "Report published."
                if report_msg:
                    try:
                        await report_msg.edit(content=content)
                    except Exception:
                        await thread.send(content)
                else:
                    await thread.send(content)
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
                await status_msg.edit(embed=discord.Embed(title="Cancelled"), view=None)
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
    await ctx.respond(f"🔬 Researching: {prompt[:100]}")
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


@bot.event
async def on_ready():
    logger.info("Logged in as %s", bot.user)


if __name__ == "__main__":
    bot.run(TOKEN)
