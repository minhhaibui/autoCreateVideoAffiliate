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
