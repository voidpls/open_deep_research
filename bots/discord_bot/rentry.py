"""Rentry.co publishing for research reports."""

import logging

logger = logging.getLogger("research-bot")

# Dark monochrome preset — Inter, flat, mobile-first
RENTRY_REPORT_METADATA = {
    "OPTION_DISABLE_VIEWS": True,
    "OPTION_DISABLE_SEARCH_ENGINE": True,
    "ACCESS_RECOMMENDED_THEME": "dark",
    "CONTAINER_PADDING": "20px",
    "CONTAINER_MAX_WIDTH": "680px",
    "CONTAINER_INNER_BACKGROUND_COLOR": "#111111",
    "CONTAINER_OUTER_BACKGROUND_COLOR": "#000000",
    "CONTAINER_BORDER_COLOR": "#262626",
    "CONTAINER_BORDER_WIDTH": "1px",
    "CONTAINER_BORDER_STYLE": "solid",
    "CONTAINER_BORDER_RADIUS": "8px",
    "CONTAINER_SHADOW_COLOR": "transparent",
    "CONTAINER_SHADOW_OFFSET": "0px",
    "CONTAINER_SHADOW_SPREAD": "0px",
    "CONTAINER_SHADOW_BLUR": "0px",
    "CONTENT_FONT": "Inter",
    "CONTENT_FONT_WEIGHT": "400 600",
    "CONTENT_TEXT_SIZE": "16px 1rem 1.9rem 1.55rem 1.3rem 1.15rem 1.05rem 1rem 1rem 1rem 1rem 0.9rem",
    "CONTENT_TEXT_COLOR": "#E5E5E5",
    "CONTENT_LINK_COLOR": "#FFFFFF",
    "CONTENT_BULLET_COLOR": "#737373",
}


def publish_to_rentry(report: str, title: str = "Research Report") -> str | None:
    """Publish markdown report to rentry.co. Returns URL or None on failure."""
    try:
        from rentry_client import RentryClient

        client = RentryClient()
        meta = {**RENTRY_REPORT_METADATA, "PAGE_TITLE": title[:60]}
        resp = client.new(text=report, metadata=meta)
        if resp.get("status") == "200":
            return resp["url"]
        logger.warning("Rentry returned status %s", resp.get("status"))
    except Exception:
        logger.warning("Rentry publish failed", exc_info=True)
    return None
