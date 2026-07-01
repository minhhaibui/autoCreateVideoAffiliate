import unittest

from app.services.onboarding import (
    EXAMPLE_CODE,
    EXAMPLE_LINK,
    EXAMPLE_PRICE,
    example_prefill,
    should_show_onboarding,
)


class TestShouldShowOnboarding(unittest.TestCase):
    def test_shown_on_blank_canvas(self):
        self.assertTrue(should_show_onboarding("", ""))
        self.assertTrue(should_show_onboarding(None, None))
        self.assertTrue(should_show_onboarding("   ", "  "))

    def test_hidden_once_subject_present(self):
        self.assertFalse(should_show_onboarding("a lamp", ""))

    def test_hidden_once_script_present(self):
        self.assertFalse(should_show_onboarding("", "This lamp changed my life."))


class TestExamplePrefill(unittest.TestCase):
    def test_uses_passed_localized_values(self):
        prefill = example_prefill("subj", "prod", "shop")
        self.assertEqual(prefill["video_subject"], "subj")
        self.assertEqual(prefill["campaign_product"], "prod")
        self.assertEqual(prefill["campaign_shop"], "shop")

    def test_includes_placeholder_campaign_fields(self):
        prefill = example_prefill("s", "p", "sh")
        self.assertEqual(prefill["campaign_price"], EXAMPLE_PRICE)
        self.assertEqual(prefill["campaign_code"], EXAMPLE_CODE)
        self.assertEqual(prefill["campaign_link"], EXAMPLE_LINK)

    def test_example_link_is_an_obvious_placeholder(self):
        # Must not look like a real affiliate link the user might ship as-is.
        self.assertIn("example.com", EXAMPLE_LINK)

    def test_keys_match_keyed_widgets(self):
        # Guards the contract with the keyed Streamlit widgets in Main.py.
        self.assertEqual(
            set(example_prefill("s", "p", "sh").keys()),
            {
                "video_subject",
                "campaign_product",
                "campaign_shop",
                "campaign_price",
                "campaign_code",
                "campaign_link",
            },
        )


if __name__ == "__main__":
    unittest.main()
