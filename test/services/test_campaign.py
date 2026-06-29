import unittest

from app.services.campaign import (
    build_campaign_section,
    format_campaign_cta,
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
