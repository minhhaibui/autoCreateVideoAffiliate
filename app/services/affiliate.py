"""Helpers for the TikTok affiliate creator toolkit that have no UI dependency.

Kept out of ``webui/Main.py`` so they can be unit-tested without importing
Streamlit (importing Main.py executes Streamlit page setup at module load).
"""


def build_affiliate_package_text(
    subject,
    script,
    keywords,
    hooks,
    social_meta,
    label,
    shots=None,
    comment_replies=None,
    sound_ideas=None,
    stickers=None,
    cover_ideas=None,
    schedule_slots=None,
):
    """Assemble all generated affiliate assets into one plain-text document the
    user can download/keep. ``label`` maps section keys to translated headings so
    the export follows the current UI language. Empty sections are skipped."""
    lines = []

    def section(title, body):
        body = (body or "").strip()
        if body:
            lines.append(f"## {title}")
            lines.append(body)
            lines.append("")

    section(label("subject"), subject)
    if hooks:
        numbered = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(hooks))
        section(label("hooks"), numbered)
    section(label("script"), script)
    if shots:
        blocks = []
        for i, shot in enumerate(shots):
            parts = [f"{i + 1}. {shot.get('scene', '')}".rstrip()]
            if shot.get("voiceover"):
                parts.append(f"   {label('shot_voiceover')}: {shot['voiceover']}")
            if shot.get("onscreen_text"):
                parts.append(f"   {label('shot_onscreen')}: {shot['onscreen_text']}")
            if shot.get("broll"):
                parts.append(f"   {label('shot_broll')}: {shot['broll']}")
            blocks.append("\n".join(parts))
        section(label("shots"), "\n".join(blocks))
    section(label("keywords"), keywords)

    if social_meta:
        section(label("title"), social_meta.get("title", ""))
        section(label("caption"), social_meta.get("caption", ""))
        hashtags = " ".join(social_meta.get("hashtags", []) or [])
        section(label("hashtags"), hashtags)

    if comment_replies:
        blocks = []
        for i, pair in enumerate(comment_replies):
            blocks.append(
                f"{i + 1}. {pair.get('comment', '')}\n"
                f"   {label('reply')}: {pair.get('reply', '')}"
            )
        section(label("comment_replies"), "\n".join(blocks))

    if sound_ideas:
        blocks = []
        for i, idea in enumerate(sound_ideas):
            parts = [f"{i + 1}. {idea.get('sound', '')}".rstrip()]
            if idea.get("vibe"):
                parts.append(f"   {label('sound_vibe')}: {idea['vibe']}")
            if idea.get("search"):
                parts.append(f"   {label('sound_search')}: {idea['search']}")
            if idea.get("tip"):
                parts.append(f"   {label('sound_tip')}: {idea['tip']}")
            blocks.append("\n".join(parts))
        section(label("sounds"), "\n".join(blocks))

    if stickers:
        blocks = []
        for i, sticker in enumerate(stickers):
            parts = [f"{i + 1}. {sticker.get('text', '')}".rstrip()]
            if sticker.get("timing"):
                parts.append(f"   {label('sticker_timing')}: {sticker['timing']}")
            if sticker.get("style"):
                parts.append(f"   {label('sticker_style')}: {sticker['style']}")
            if sticker.get("purpose"):
                parts.append(f"   {label('sticker_purpose')}: {sticker['purpose']}")
            blocks.append("\n".join(parts))
        section(label("stickers"), "\n".join(blocks))

    if cover_ideas:
        blocks = []
        for i, idea in enumerate(cover_ideas):
            parts = [f"{i + 1}. {idea.get('text', '')}".rstrip()]
            if idea.get("subtext"):
                parts.append(f"   {label('cover_subtext')}: {idea['subtext']}")
            if idea.get("angle"):
                parts.append(f"   {label('cover_angle')}: {idea['angle']}")
            if idea.get("tip"):
                parts.append(f"   {label('cover_tip')}: {idea['tip']}")
            blocks.append("\n".join(parts))
        section(label("covers"), "\n".join(blocks))

    if schedule_slots:
        blocks = []
        for i, slot in enumerate(schedule_slots):
            head = " — ".join(
                p for p in [slot.get("slot", ""), slot.get("time", "")] if p
            )
            parts = [f"{i + 1}. {head}".rstrip()]
            if slot.get("day"):
                parts.append(f"   {label('schedule_day')}: {slot['day']}")
            if slot.get("why"):
                parts.append(f"   {label('schedule_why')}: {slot['why']}")
            blocks.append("\n".join(parts))
        section(label("schedule"), "\n".join(blocks))

    return "\n".join(lines).strip() + "\n"
