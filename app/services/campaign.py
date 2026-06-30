"""Affiliate campaign helpers — the single source of truth for one product's
exact affiliate link, discount code, price and shop.

These values must be reproduced VERBATIM (an affiliate link or discount code that
the LLM paraphrases is worthless or, worse, wrong), so nothing here ever touches
the language model: all formatting is deterministic and unit-tested. The same
campaign feeds the export bundle, the pinned-comment / caption CTA and — later —
the on-screen CTA burned into the video, so the link and code can never drift
between surfaces.
"""

# Order matters: this is also the top-to-bottom order fields render in.
CAMPAIGN_FIELDS = ("product", "price", "code", "link", "shop")


def normalize_campaign(raw) -> dict:
    """Return every campaign field as a trimmed string, with missing keys as ''."""
    raw = raw or {}
    return {key: str(raw.get(key, "") or "").strip() for key in CAMPAIGN_FIELDS}


def has_campaign(campaign) -> bool:
    """True when at least one campaign field carries a value."""
    data = normalize_campaign(campaign)
    return any(data[key] for key in CAMPAIGN_FIELDS)


def format_campaign_cta(campaign, label) -> str:
    """A ready-to-paste call-to-action block built ONLY from the user's exact
    values — link and code are copied verbatim, never generated. Empty fields are
    skipped. ``label`` maps keys to translated words so the CTA follows the UI
    language. Returns '' when the campaign is empty.

    Shaped for a TikTok pinned comment / caption: emoji anchors + short lines.
    """
    data = normalize_campaign(campaign)
    lines = []

    if data["product"]:
        head = data["product"]
        if data["price"]:
            head += f" — {data['price']}"
        lines.append(f"🛒 {head}")
    elif data["price"]:
        lines.append(f"🛒 {data['price']}")

    if data["code"]:
        lines.append(f"🎁 {label('campaign_code_label')}: {data['code']}")
    if data["link"]:
        lines.append(f"👉 {label('campaign_link_label')}: {data['link']}")
    if data["shop"]:
        lines.append(f"🏪 {data['shop']}")

    return "\n".join(lines)


def format_onscreen_cta(campaign, label) -> str:
    """A short, PLAIN-TEXT CTA to burn onto the video — no emoji, because the
    bundled subtitle fonts have no emoji glyphs and would render boxes. Shows the
    discount code (verbatim) when present, plus a pointer to the pinned-comment
    link. Returns '' when the campaign is empty. Used only as the editable
    default; the user controls the exact burned text."""
    data = normalize_campaign(campaign)
    if not has_campaign(data):
        return ""
    lines = []
    if data["code"]:
        lines.append(f"{label('campaign_code_label')}: {data['code']}")
    lines.append(label("campaign_onscreen_pointer"))
    return "\n".join(lines)


def format_end_card_text(campaign, label) -> str:
    """A short, PLAIN-TEXT closing 'end card' to append after the video — no emoji,
    same reason as the burned CTA: the bundled fonts have no emoji glyphs. Stacks
    the product, price, discount code (verbatim) and a pointer to the pinned-comment
    link, each on its own line, to be shown large and centered on a solid screen.
    Returns '' when the campaign is empty. Used only as the editable default; the
    user controls the exact end-card text."""
    data = normalize_campaign(campaign)
    if not has_campaign(data):
        return ""
    lines = []
    if data["product"]:
        lines.append(data["product"])
    if data["price"]:
        lines.append(data["price"])
    if data["code"]:
        lines.append(f"{label('campaign_code_label')}: {data['code']}")
    lines.append(label("campaign_onscreen_pointer"))
    return "\n".join(lines)


def build_campaign_section(campaign, label) -> str:
    """A plain-text details block for the export bundle: every present field on
    its own labelled line. Returns '' when the campaign is empty."""
    data = normalize_campaign(campaign)
    rows = [
        ("campaign_product_label", data["product"]),
        ("campaign_price_label", data["price"]),
        ("campaign_code_label", data["code"]),
        ("campaign_link_label", data["link"]),
        ("campaign_shop_label", data["shop"]),
    ]
    lines = [f"{label(key)}: {value}" for key, value in rows if value]
    return "\n".join(lines)
