"""Autopilot: unattended product-pick + video render, meant to run from cron.

One run = one video: generate product ideas inside the account's channel
niche, rank them by conversion potential for the stock-footage format, skip
products already rendered recently (history file), render the winner through
the normal task pipeline, then generate the publish copy (caption + pinned
comment) and drop everything into a plain-text report next to the video.

Kept UI-free so a cron entry can call ``autopilot.py`` at the repo root
without Streamlit; every setting comes from config.toml (the same
``channel_niche`` and voice/subtitle settings the WebUI uses).
"""

import json
import os
import time
from datetime import datetime

from loguru import logger

from app.config import config
from app.models.schema import VideoParams
from app.services import llm
from app.services import task as tm
from app.services.preflight import is_llm_ready
from app.utils import utils

# Products rendered within the last N runs are skipped so back-to-back cron
# ticks don't produce near-duplicate videos for the same account.
HISTORY_LIMIT = 20

# How many candidate ideas the picker generates before ranking.
IDEA_POOL_SIZE = 8

REPORT_FILENAME = "autopilot_report.txt"

# Free-tier Gemini allows ~5 requests/minute and the shared JSON retry loop
# burns them in seconds. An unattended run has nowhere to hurry to, so it
# pauses between LLM stages and, when a stage comes back empty (usually a
# drained quota window), waits the window out and tries the stage again.
LLM_COOLDOWN_SECONDS = 30
QUOTA_RETRY_WAIT_SECONDS = 70
PICK_ATTEMPTS = 3


def _cooldown(seconds: float = LLM_COOLDOWN_SECONDS) -> None:
    time.sleep(seconds)


def autopilot_dir() -> str:
    return utils.storage_dir("autopilot", create=True)


def history_path() -> str:
    return os.path.join(autopilot_dir(), "history.json")


def load_history(path: str) -> list:
    """Read the used-product history; corrupt or missing files mean 'empty'
    rather than a crashed cron run."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
        return history if isinstance(history, list) else []
    except (OSError, ValueError):
        return []


def save_history(path: str, history: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history[-HISTORY_LIMIT:], f, ensure_ascii=False, indent=2)


def pick_best_product(
    niche: str,
    language: str = "vi",
    history: list = None,
    pool_size: int = IDEA_POOL_SIZE,
) -> dict:
    """Generate a pool of in-niche product ideas and return the most promising
    unused one (idea dict + a "why_best" sentence from the ranking pass).

    Ideas whose product name was already rendered (history) are skipped; if
    every candidate was used the freshest pool is allowed again rather than
    failing the run. Falls back to the pool's original order when the ranking
    pass returns nothing usable. Returns None only when no ideas came back.
    """
    ideas = []
    for attempt in range(PICK_ATTEMPTS):
        if attempt:
            logger.warning(
                f"autopilot: empty idea pool (likely a drained quota window), "
                f"waiting {QUOTA_RETRY_WAIT_SECONDS}s before attempt {attempt + 1}"
            )
            _cooldown(QUOTA_RETRY_WAIT_SECONDS)
        ideas = [
            i
            for i in llm.generate_product_ideas(
                category=niche,
                market="Vietnam",
                language=language,
                amount=pool_size,
            )
            if (i or {}).get("product")
        ]
        if ideas:
            break
    if not ideas:
        return None

    used = {(h.get("product") or "").casefold() for h in (history or [])}
    fresh = [i for i in ideas if i["product"].casefold() not in used]
    if not fresh:
        logger.warning(
            "autopilot: every candidate was rendered recently, allowing repeats"
        )
        fresh = ideas

    _cooldown()
    ranked = llm.rank_product_ideas(fresh, niche=niche, language=language)
    for entry in ranked:
        name = (entry.get("product") or "").casefold()
        for idea in fresh:
            if idea["product"].casefold() == name:
                return {**idea, "why_best": entry.get("why", "")}
    # Ranking failed or echoed names we can't match — the pool order is the
    # model's own implicit preference, so its head is a sane winner.
    return {**fresh[0], "why_best": ""}


def build_video_params(subject: str, language: str = "vi") -> VideoParams:
    """One portrait video from an empty script (the pipeline generates script
    and search terms per subject), voiced and subtitled with the same config.ui
    settings the WebUI run would use."""
    ui = config.ui
    return VideoParams(
        video_subject=subject,
        video_script="",
        video_terms="",
        video_language=language,
        video_source="pexels",
        video_count=1,
        video_clip_duration=4,
        paragraph_number=1,
        voice_name=ui.get("voice_name", "") or "vi-VN-HoaiMyNeural-Female",
        voice_rate=1.0,
        subtitle_enabled=True,
        font_name=ui.get("font_name", "") or "BeVietnamPro-Bold.ttf",
        text_fore_color=ui.get("text_fore_color", "#FFFFFF"),
        font_size=int(ui.get("font_size", 60) or 60),
        subtitle_position=ui.get("subtitle_position", "bottom"),
    )


def format_report(product: dict, videos: list, metadata: dict, pinned: list) -> str:
    """Deterministic plain-text bundle: what was rendered, where, and the
    ready-to-paste publish copy. Mirrors the batch report's shape."""
    lines = [
        f"Autopilot run: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Product: {product.get('product', '')}",
    ]
    if product.get("why_best"):
        lines.append(f"Why picked: {product['why_best']}")
    if product.get("angle"):
        lines.append(f"Angle: {product['angle']}")
    for v in videos:
        lines.append(f"Video: {v}")
    metadata = metadata or {}
    if metadata.get("title"):
        lines.append(f"Title: {metadata['title']}")
    if metadata.get("caption"):
        lines.append(f"Caption: {metadata['caption']}")
    hashtags = [h for h in (metadata.get("hashtags") or []) if h]
    if hashtags:
        lines.append("Hashtags: " + " ".join(hashtags))
    first_pinned = (pinned[0] if pinned else {}) or {}
    if first_pinned.get("comment"):
        lines.append(f"Pinned: {first_pinned['comment']}")
    lines.append("NOTE: add your real affiliate link to the pinned comment "
                 "before publishing — autopilot never invents links.")
    return "\n".join(lines)


def run_autopilot(niche: str = "", language: str = "vi") -> dict:
    """One full unattended cycle. Returns
    ``{"product", "task_id", "videos", "report", "error"}`` where ``videos``
    is empty and ``error`` set on failure — the CLI turns that into the exit
    code. Publish-copy failures never fail a finished render (the video is
    the expensive part); they are logged and the report just omits the copy.
    """
    niche = (niche or config.app.get("channel_niche", "")).strip()
    if not niche:
        return {"error": "no niche: set channel_niche in config.toml [app] "
                         "or pass --niche", "videos": []}

    provider = config.app.get("llm_provider", "").lower()
    if not is_llm_ready(provider, config.app.get(f"{provider}_api_key", "")):
        return {"error": f"llm not ready (provider={provider})", "videos": []}
    if not config.app.get("pexels_api_keys", ""):
        return {"error": "no pexels api key", "videos": []}

    history = load_history(history_path())
    product = pick_best_product(niche, language=language, history=history)
    if not product:
        return {"error": "no product ideas returned", "videos": []}
    subject = product["product"]
    logger.info(f"autopilot picked: {subject!r} ({product.get('why_best', '')})")

    task_id = utils.get_uuid()
    params = build_video_params(subject, language=language)
    # the render pipeline opens with two more LLM calls (script + terms)
    _cooldown()
    result = tm.start(task_id=task_id, params=params)
    videos = (result or {}).get("videos") or []
    if not videos:
        return {
            "error": f"render failed for {subject!r}",
            "product": subject,
            "task_id": task_id,
            "videos": [],
        }

    metadata, pinned = {}, []
    try:
        metadata = llm.generate_social_metadata(
            video_subject=subject,
            language=language,
            platform="tiktok",
            niche=niche,
        )
        _cooldown()
        pinned = llm.generate_pinned_comments(
            video_subject=subject, language=language, amount=1
        )
    except Exception as e:
        logger.warning(f"autopilot: publish copy failed, keeping the video: {e}")

    report = format_report(product, videos, metadata, pinned)
    report_path = os.path.join(utils.task_dir(task_id), REPORT_FILENAME)
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
    except OSError as e:
        logger.warning(f"autopilot: could not write report: {e}")
        report_path = ""

    history.append(
        {
            "product": subject,
            "task_id": task_id,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
    )
    save_history(history_path(), history)

    return {
        "product": subject,
        "task_id": task_id,
        "videos": videos,
        "report": report_path,
        "error": "",
    }
