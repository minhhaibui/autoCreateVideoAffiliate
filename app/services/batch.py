"""Batch-mode helpers, with no UI dependency.

Kept out of ``webui/Main.py`` so the "turn a pasted product list into render
jobs" logic can be unit-tested without importing Streamlit. The WebUI batch
panel lets an affiliate creator paste one product/subject per line and render
a video for each in one go, reusing the current voice/source/subtitle settings;
each item gets its own LLM-generated script, so the batch needs a working LLM
provider regardless of the script box's contents.
"""

import re

# Rendering is minutes per video, so the cap keeps a pasted catalog from
# turning into an hours-long surprise; the UI says how many lines were kept.
MAX_BATCH_ITEMS = 10

# Leading list decorations people paste from notes or spreadsheets:
# "1. ", "2)", "- ", "* ", "• ".
_LIST_PREFIX = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•])\s+")

# An affiliate link is the one extra we can spot with certainty.
_LINK_RE = re.compile(r"^(?:https?://|www\.)", re.IGNORECASE)

# A price is digits with separators plus an optional currency-ish tail
# ("199k", "1.299.000đ", "$15.99", "15.99 USD", "30%"). The tail is a
# whitelist on purpose: "50OFF" must stay a discount code, not a price.
_PRICE_RE = re.compile(
    r"^[$€₫]?\s*\d[\d.,\s]*\s*(?:k|tr|đ|d|vnd|vnđ|usd|eur|%|\$|€|₫)?$",
    re.IGNORECASE,
)

# Per-line campaign fields a batch item may carry beyond the subject itself.
BATCH_EXTRA_FIELDS = ("price", "code", "link")


def _classify_extra(part):
    """Map one ``|``-separated extra to the campaign field it looks like."""
    if _LINK_RE.match(part):
        return "link"
    if _PRICE_RE.match(part):
        return "price"
    return "code"


def parse_batch_items(text, max_items=MAX_BATCH_ITEMS):
    """Turn one-product-per-line text into capped, deduped batch items.

    Each line is ``subject`` alone or ``subject | extras...`` where every
    extra is recognized by shape — a URL becomes ``link``, digits with an
    optional currency tail become ``price``, anything else becomes ``code``
    — so the user can paste extras in any order. The first match per field
    wins. Lines are stripped of whitespace and common list prefixes, empty
    lines are dropped, and subjects are deduped case-insensitively while
    preserving order. Returns a list of
    ``{"subject", "price", "code", "link"}`` dicts (missing extras are '').
    """
    items = []
    seen = set()
    for line in (text or "").splitlines():
        parts = [p.strip() for p in _LIST_PREFIX.sub("", line).split("|")]
        subject = parts[0]
        if not subject:
            continue
        fingerprint = subject.casefold()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        item = {"subject": subject, "price": "", "code": "", "link": ""}
        for part in parts[1:]:
            if not part:
                continue
            field = _classify_extra(part)
            if not item[field]:
                item[field] = part
        items.append(item)
    return items[:max_items]


def has_batch_extras(item):
    """True when a batch item carries any per-line campaign field."""
    return any(item.get(field) for field in BATCH_EXTRA_FIELDS)


def parse_batch_subjects(text, max_items=MAX_BATCH_ITEMS):
    """Turn one-subject-per-line text into a clean, capped list of subjects.

    Same parsing as :func:`parse_batch_items`, keeping only the subjects.
    """
    return [item["subject"] for item in parse_batch_items(text, max_items)]


def summarize_batch_results(results):
    """Build a plain-text batch report from per-item result dicts.

    Each item is ``{"subject": str, "videos": [paths], "error": str}`` where
    ``videos`` is empty on failure and ``error`` is empty on success. An item
    may also carry ``"cta"`` — a ready-to-paste pinned-comment block built
    from that line's own campaign fields — which is indented under the item
    so the link and code travel with the report verbatim. The summary is
    deterministic so it can be shown and exported as-is.
    """
    lines = []
    ok = sum(1 for r in results if r.get("videos"))
    lines.append(f"{ok}/{len(results)} OK")
    for i, r in enumerate(results, start=1):
        subject = r.get("subject", "")
        if r.get("videos"):
            lines.append(f"{i}. [OK] {subject}")
            for v in r["videos"]:
                lines.append(f"   - {v}")
        else:
            reason = r.get("error") or "failed"
            lines.append(f"{i}. [FAILED] {subject} ({reason})")
        for cta_line in (r.get("cta") or "").splitlines():
            lines.append(f"   {cta_line}")
    return "\n".join(lines)
