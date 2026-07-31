# Research Bot for Discord

A Discord bot for deep research. Run `/research <question>` and it asks clarifying questions, shows you a research plan, posts progress updates, and delivers a report, all in a thread.

Built on a vendored copy of [open_deep_research](https://github.com/langchain-ai/open_deep_research). Bot code lives in `api.py` and `bots/discord_bot/`.

## Usage

1. `/research <question>` opens a thread.
2. The bot may ask a clarifying question. Reply in the thread.
3. It shows the research brief. Press Start.
4. A status message updates while it researches.
5. When finished: a rentry.co link with dark-mode styled report, plus the full report as a markdown attachment.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra bot
cp .env.example .env
```

Fill in `.env`:

| Variable | Purpose |
|---|---|
| `DISCORD_BOT_TOKEN` | Bot token from the [Discord developer portal](https://discord.com/developers/applications) |
| `TAVILY_API_KEY` | Web search. Free tier at [tavily.com](https://www.tavily.com/) is enough |
| `OPENAI_API_KEY` | Any OpenAI-compatible endpoint |
| `OPENAI_BASE_URL` | Only needed for a non-OpenAI bridge or gateway |
| `GROQ_API_KEY` | Summarizes research topics into one-line headers. Free tier at [groq.com](https://console.groq.com/) |

Invite the bot with the `bot` and `applications.commands` scopes. Enable the **Message Content** privileged intent in the [Discord developer portal](https://discord.com/developers/applications) (Bot → Privileged Gateway Intents) — required for the bot to read clarification messages in threads.

Either give bot "Create Public Threads" permission, or invite your bot with this link (replace YOUR_BOT_ID):

https://discord.com/oauth2/authorize?client_id=YOUR_BOT_ID&permissions=34359738368&integration_type=0&scope=bot+applications.commands

```bash
uv run python bots/discord_bot/bot.py
```

## Configuration

Model assignments live in the `MATRIX` dict at the top of `api.py`. The defaults are designed around [OpenCode Go](https://opencode.ai) models, but any OpenAI- or Anthropic-compatible model works (`openai:` / `anthropic:` prefix):

```python
# api.py
MATRIX = {
    "structured_model":    "openai:mimo-v2.5-pro",  # clarification, plan
    "research_model":      "openai:minimax-m3",     # research agents
    "summarization_model": "openai:mimo-v2.5",      # search result summaries
    "compression_model":   "openai:mimo-v2.5",      # research compression
    "final_report_model":  "openai:minimax-m3",     # report writing
}
```

Edit that dict to change models. `OPENAI_BASE_URL` in `.env` controls which endpoint the `openai:` models hit.

## The research engine

`src/open_deep_research/` is [open_deep_research](https://github.com/langchain-ai/open_deep_research) from LangChain: a LangGraph agent that clarifies, plans, runs parallel researchers, and writes a report. See the upstream repo for its docs and evaluation harness. `bots/discord_bot/PLAN.md` has the bot spec and notes on how the two connect.

To run the engine without Discord, via LangGraph Studio:

```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```
