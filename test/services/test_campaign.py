import unittest

from app.services.campaign import (
    build_campaign_section,
    format_campaign_cta,
    format_end_card_text,
    format_onscreen_cta,
    has_campaign,
    normalize_campaign,
)


def _label(key):
    """Identity-ish label: returns the raw key so assertions are explicit."""
    return key


FULL = {
    "product": "Mini portable blender",
    "price": "199k",
    "code": "SALE10",
    "link": "https://shp.ee/abc123",
    "shop": "MyShop Official",
}


class TestNormalizeCampaign(unittest.TestCase):
    def test_trims_and_fills_missing(self):
        out = normalize_campaign({"product": "  Blender  ", "price": "199k"})
        self.assertEqual(out["product"], "Blender")
        self.assertEqual(out["price"], "199k")
        self.assertEqual(out["code"], "")
        self.assertEqual(out["link"], "")
        self.assertEqual(out["shop"], "")

    def test_none_is_all_empty(self):
        self.assertEqual(
            normalize_campaign(None),
            {"product": "", "price": "", "code": "", "link": "", "shop": ""},
        )

    def test_non_string_values_are_stringified(self):
        self.assertEqual(normalize_campaign({"price": 199})["price"], "199")


class TestHasCampaign(unittest.TestCase):
    def test_empty(self):
        self.assertFalse(has_campaign(None))
        self.assertFalse(has_campaign({}))
        self.assertFalse(has_campaign({"product": "   "}))

    def test_any_field_present(self):
        self.assertTrue(has_campaign({"link": "https://x"}))
        self.assertTrue(has_campaign(FULL))


class TestFormatCampaignCta(unittest.TestCase):
    def test_empty_returns_blank(self):
        self.assertEqual(format_campaign_cta(None, _label), "")

    def test_link_and_code_are_verbatim(self):
        cta = format_campaign_cta(FULL, _label)
        self.assertIn("https://shp.ee/abc123", cta)
        self.assertIn("SALE10", cta)

    def test_product_and_price_joined(self):
        cta = format_campaign_cta(FULL, _label)
        self.assertIn("🛒 Mini portable blender — 199k", cta)

    def test_skips_missing_fields(self):
        cta = format_campaign_cta({"link": "https://x"}, _label)
        self.assertEqual(cta, "👉 campaign_link_label: https://x")

    def test_price_only_still_shows_cart_line(self):
        self.assertIn("🛒 199k", format_campaign_cta({"price": "199k"}, _label))


class TestFormatOnscreenCta(unittest.TestCase):
    def test_empty_returns_blank(self):
        self.assertEqual(format_onscreen_cta(None, _label), "")
        self.assertEqual(format_onscreen_cta({}, _label), "")

    def test_has_no_emoji(self):
        cta = format_onscreen_cta(FULL, _label)
        self.assertTrue(cta.isascii() or all(ord(c) < 0x1F000 for c in cta))
        # explicit: none of the emoji used by the pasteable CTA leak in
        for emoji in ("🛒", "🎁", "👉", "🏪", "👇"):
            self.assertNotIn(emoji, cta)

    def test_includes_code_verbatim_and_pointer(self):
        cta = format_onscreen_cta(FULL, _label)
        self.assertIn("SALE10", cta)
        self.assertIn("campaign_onscreen_pointer", cta)

    def test_pointer_only_when_no_code(self):
        cta = format_onscreen_cta({"product": "Blender"}, _label)
        self.assertEqual(cta, "campaign_onscreen_pointer")


class TestFormatEndCardText(unittest.TestCase):
    def test_empty_returns_blank(self):
        self.assertEqual(format_end_card_text(None, _label), "")
        self.assertEqual(format_end_card_text({}, _label), "")

    def test_has_no_emoji(self):
        text = format_end_card_text(FULL, _label)
        for emoji in ("🛒", "🎁", "👉", "🏪", "👇"):
            self.assertNotIn(emoji, text)

    def test_stacks_product_price_code_and_pointer(self):
        text = format_end_card_text(FULL, _label)
        self.assertEqual(
            text.splitlines(),
            [
                "Mini portable blender",
                "199k",
                "campaign_code_label: SALE10",
                "campaign_onscreen_pointer",
            ],
        )

    def test_code_is_verbatim(self):
        text = format_end_card_text({"code": "Save-20%!"}, _label)
        self.assertIn("Save-20%!", text)

    def test_pointer_only_when_no_fields_but_one_present(self):
        text = format_end_card_text({"product": "Blender"}, _label)
        self.assertEqual(text.splitlines(), ["Blender", "campaign_onscreen_pointer"])


class TestBuildCampaignSection(unittest.TestCase):
    def test_empty_returns_blank(self):
        self.assertEqual(build_campaign_section(None, _label), "")

    def test_lists_present_fields_in_order(self):
        out = build_campaign_section(FULL, _label)
        self.assertEqual(
            out.splitlines(),
            [
                "campaign_product_label: Mini portable blender",
                "campaign_price_label: 199k",
                "campaign_code_label: SALE10",
                "campaign_link_label: https://shp.ee/abc123",
                "campaign_shop_label: MyShop Official",
            ],
        )

    def test_skips_missing_fields(self):
        out = build_campaign_section({"link": "https://x"}, _label)
        self.assertEqual(out, "campaign_link_label: https://x")


if __name__ == "__main__":
    unittest.main()
