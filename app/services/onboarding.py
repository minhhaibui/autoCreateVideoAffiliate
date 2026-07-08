"""First-run onboarding helpers, with no UI dependency.

Kept out of ``webui/Main.py`` so the "should we greet a brand-new user?" logic
and the demo prefill can be unit-tested without importing Streamlit. The WebUI
shows a getting-started strip with a one-click "Load example" so a non-technical
creator sees a fully populated flow instead of a wall of empty controls.
"""

# Illustrative campaign placeholders for the demo prefill. These are obviously
# not real (example.com is the reserved placeholder domain) so the user knows to
# replace them with their own affiliate link / code / price — matching the
# toolkit's rule that the real link/code/price is never invented for them.
EXAMPLE_PRICE = "199.000đ"
EXAMPLE_CODE = "DEMO10"
EXAMPLE_LINK = "https://shop.example.com/demo-product"


def should_show_onboarding(video_subject, video_script):
    """Return True only on a blank canvas (no subject and no script yet).

    The strip is self-dismissing: once the user types a subject or loads the
    example, it stops showing.
    """
    return not (video_subject or "").strip() and not (video_script or "").strip()


def example_prefill(subject, product, shop):
    """Build the demo prefill mapping widget-key -> value.

    ``subject``, ``product`` and ``shop`` are passed in already localized by the
    caller; price / code / link are illustrative placeholders the user replaces.
    Keys match the keyed Streamlit widgets in ``webui/Main.py``.
    """
    return {
        "video_subject": subject,
        "campaign_product": product,
        "campaign_shop": shop,
        "campaign_price": EXAMPLE_PRICE,
        "campaign_code": EXAMPLE_CODE,
        "campaign_link": EXAMPLE_LINK,
    }


# One product's per-video content, cleared by the WebUI's "New video" button so
# the canvas is ready for the next product. Text/keyed-widget fields are blanked
# (so the widgets re-read ""); generated-asset panels are removed entirely.
NEW_VIDEO_TEXT_KEYS = (
    "video_subject",
    "video_script",
    "video_terms",
    "campaign_product",
    "campaign_price",
    "campaign_code",
    # per-product: every TikTok Shop product has its own affiliate link, so it
    # must NOT carry over to the next video (shipping product A's link on
    # product B's video would be the worst kind of silent error).
    "campaign_link",
    "hook_text_input",
    "onscreen_cta_text",
    "end_card_text",
)
NEW_VIDEO_ASSET_KEYS = (
    "video_hooks",
    "video_shots",
    "script_variants",
    "social_metadata",
    "comment_replies",
    "pinned_comments",
    "disclosure_lines",
    "save_share_prompts",
    "buyer_qa",
    "schedule_slots",
    "performance_insights",
    "sound_ideas",
    "text_stickers",
    "cover_ideas",
    "last_video_result",
)

# Account-level / cross-video state that "New video" must NEVER clear: the
# channel niche, the shop name (one shop per account), the batch list, and the
# multi-video planning tools. NOT the affiliate link — that is per-product and
# is cleared above. Listed explicitly so a unit test can prove the reset leaves
# them alone even if new keys are added later.
NEW_VIDEO_PRESERVED_KEYS = (
    "channel_niche_input",
    "campaign_shop",
    "batch_subjects",
    "product_ideas",
    "content_calendar",
    "ui_language",
)


def apply_new_video_reset(state):
    """Clear one product's per-video content in ``state`` (a dict-like session
    state) in place: blank the text/widget fields, drop the generated-asset
    panels. Account-level state, the batch list and the planning tools are left
    untouched. Pure (no Streamlit dependency) so it is unit-testable."""
    for key in NEW_VIDEO_TEXT_KEYS:
        state[key] = ""
    for key in NEW_VIDEO_ASSET_KEYS:
        state.pop(key, None)


def fill_empty_product_from_subject(state):
    """Mirror the video subject into an EMPTY campaign product in ``state`` (in
    place): the two are the same thing typed in two boxes, so a filled subject
    should seed the product name that the CTA / end-card / export read. Never
    overwrites a product the user already typed. Pure, so it is unit-testable."""
    subject = (state.get("video_subject") or "").strip()
    if subject and not (state.get("campaign_product") or "").strip():
        state["campaign_product"] = subject
