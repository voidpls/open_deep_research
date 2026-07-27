"""Rentry.co publishing for research reports."""

import logging

logger = logging.getLogger("research-bot")

# Dark monochrome preset — Inter, flat, mobile-first
RENTRY_REPORT_METADATA = {
    "OPTION_DISABLE_VIEWS": True,
    "OPTION_DISABLE_SEARCH_ENGINE": True,
    "ACCESS_RECOMMENDED_THEME": "dark",
    "CONTAINER_PADDING": "16px",
    "CONTAINER_MAX_WIDTH": "640px",
    "CONTAINER_INNER_BACKGROUND_COLOR": "#0C0C0C",
    "CONTAINER_OUTER_BACKGROUND_COLOR": "#000000",
    "CONTAINER_BORDER_COLOR": "#1A1A1A",
    "CONTAINER_BORDER_WIDTH": "1px",
    "CONTAINER_BORDER_STYLE": "solid",
    "CONTAINER_BORDER_RADIUS": "6px",
    "CONTAINER_SHADOW_COLOR": "transparent",
    "CONTAINER_SHADOW_OFFSET": "0px",
    "CONTAINER_SHADOW_SPREAD": "0px",
    "CONTAINER_SHADOW_BLUR": "0px",
    "CONTENT_FONT": "IBM_Plex_Sans",
    "CONTENT_FONT_WEIGHT": "400 600",
    "CONTENT_TEXT_SIZE": "16px 1rem 1.75rem 1.4rem 1.2rem 1.1rem 1.05rem 1rem 1rem 1rem 1.05rem 0.875rem",
    "CONTENT_TEXT_COLOR": "#E5E5E5",
    "CONTENT_LINK_COLOR": "#A3A3A3",
    "CONTENT_BULLET_COLOR": "#525252",
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
        logger.warning("Rentry error (status %s): %s", resp.get("status"), resp.get("errors", resp.get("content", "unknown")))
    except Exception:
        logger.warning("Rentry publish failed", exc_info=True)
    return None
