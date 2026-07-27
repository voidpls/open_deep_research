#!/usr/bin/env python3
"""Discord research bot — /research slash command, thread-per-run."""

import asyncio
import logging
import os
import sys
from pathlib import Path

import discord
from dotenv import load_dotenv

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

load_dotenv()
TOKEN = os.environ["DISCORD_BOT_TOKEN"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("research-bot")

bot = discord.Bot()

# Per-user gate: user_id -> (task, thread)
_active: dict[int, tuple[asyncio.Task, discord.Thread]] = {}
_MAX_ITER = 3

_GLYPH = {
    "clarify": "📝",
    "brief": "⏸",
    "research": "🔬",
    "report": "✍️",
}


def _fmt_status(s: dict) -> str:
    g = _GLYPH.get(s["phase"], "🔬")
    p = s["phase"]
    if p == "clarify":
        return f"{g} Clarifying..."
    if p == "brief":
        return f"{g} Awaiting confirmation..."
    if p == "report":
        return f"{g} Writing report..."
    cur, tot = s["iteration"]
    parts = [f"{g} iter {cur}/{tot}"]
    if s["researchers"]:
        parts.append(f"{s['researchers']} researchers")
    if s["sources"]:
        parts.append(f"{s['sources']} sources")
    return " · ".join(parts)


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


async def _run(bot, thread: discord.Thread, uid: int, prompt: str):
    inbox = asyncio.Queue()
    cancelled = asyncio.Event()

    state = {
        "phase": "clarify",
        "iteration": (0, _MAX_ITER),
        "researchers": 0,
        "sources": 0,
    }

    status_msg = await thread.send("📝 Clarifying...")

    # --- Cancel button (lives on status msg for whole run) ---
    cancel_view = discord.ui.View(timeout=None)

    async def _on_cancel(interaction: discord.Interaction):
        cancelled.set()
        await inbox.put("cancel")
        try:
            await interaction.response.edit_message(content="❌ Cancelled", view=None)
        except Exception:
            pass

    cb = discord.ui.Button(label="Cancel", style=discord.ButtonStyle.danger)
    cb.callback = _on_cancel
    cancel_view.add_item(cb)

    # --- Periodic status updater ---
    async def _upd():
        while not cancelled.is_set():
            try:
                await status_msg.edit(content=_fmt_status(state), view=cancel_view)
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
                state["phase"] = "clarify"
                await thread.send(event.question)

                def _check(m):
                    return m.channel.id == thread.id and m.author.id == uid

                # Race wait_for against cancel button
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
                    # Cancel was clicked
                    await inbox.put("cancel")
                    break

                try:
                    reply = wt.result()
                    await inbox.put(reply.content)
                except asyncio.TimeoutError:
                    await inbox.put("cancel")
                    await thread.send("❌ Timed out — cancelled.")
                    break

            # ----- BriefReady -----
            elif isinstance(event, BriefReady):
                state["phase"] = "brief"

                bv = discord.ui.View(timeout=600)

                async def _start(interaction: discord.Interaction):
                    await inbox.put("start")
                    await interaction.response.edit_message(view=None)

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
                        await bv.message.edit(content="⏸ Timed out — cancelled.", view=None)
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
                # API handles inbox.get() — bot only puts via button callbacks

            # ----- SupervisorTick -----
            elif isinstance(event, SupervisorTick):
                state["phase"] = "research"
                state["iteration"] = (event.research_iterations, _MAX_ITER)

            # ----- SupervisorToolsDone -----
            elif isinstance(event, SupervisorToolsDone):
                state["phase"] = "research"
                state["researchers"] = event.researchers_spawned
                state["sources"] = event.new_urls

            # ----- ReportStarted -----
            elif isinstance(event, ReportStarted):
                state["phase"] = "report"

            # ----- Done -----
            elif isinstance(event, Done):
                cancelled.set()
                upd_task.cancel()
                await status_msg.edit(content="✅ Done", view=None)
                summary = _trunc(event.report, 1800)
                await thread.send(summary)
                await thread.send(file=discord.File(event.report_path))
                return

            # ----- Failed -----
            elif isinstance(event, Failed):
                cancelled.set()
                upd_task.cancel()
                logger.error("Research failed:\n%s", event.error)
                await status_msg.edit(content=f"❌ {event.error}", view=None)
                return

    except asyncio.CancelledError:
        cancelled.set()
        upd_task.cancel()
        try:
            await status_msg.edit(content="❌ Cancelled", view=None)
        except Exception:
            pass
        raise
    finally:
        cancelled.set()
        if not upd_task.done():
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

    await ctx.defer()

    thread_name = f"🔬 {prompt[:40]}"
    start_msg = await ctx.send(f"🔬 Researching: {prompt[:100]}")
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
