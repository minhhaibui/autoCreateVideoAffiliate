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


def parse_batch_subjects(text, max_items=MAX_BATCH_ITEMS):
    """Turn one-subject-per-line text into a clean, capped list of subjects.

    Strips whitespace and common list prefixes, drops empty lines, dedupes
    case-insensitively while preserving order, and keeps at most ``max_items``
    entries. Returns a list of subject strings.
    """
    subjects = []
    seen = set()
    for line in (text or "").splitlines():
        subject = _LIST_PREFIX.sub("", line).strip()
        if not subject:
            continue
        fingerprint = subject.casefold()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        subjects.append(subject)
    return subjects[:max_items]


def summarize_batch_results(results):
    """Build a plain-text batch report from per-item result dicts.

    Each item is ``{"subject": str, "videos": [paths], "error": str}`` where
    ``videos`` is empty on failure and ``error`` is empty on success. The
    summary is deterministic so it can be shown and exported verbatim.
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
    return "\n".join(lines)
