import unittest

from app.services.onboarding import (
    EXAMPLE_CODE,
    EXAMPLE_LINK,
    EXAMPLE_PRICE,
    NEW_VIDEO_ASSET_KEYS,
    NEW_VIDEO_PRESERVED_KEYS,
    NEW_VIDEO_TEXT_KEYS,
    apply_new_video_reset,
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


class TestApplyNewVideoReset(unittest.TestCase):
    def _full_canvas(self):
        """A session state carrying one product's per-video content AND the
        account-level / planning state that must survive a reset."""
        state = {k: f"val-{k}" for k in NEW_VIDEO_TEXT_KEYS}
        state.update({k: [f"item-{k}"] for k in NEW_VIDEO_ASSET_KEYS})
        state.update(
            {
                "channel_niche_input": "đồ gia dụng nhà bếp",
                "campaign_shop": "MyShop",
                "campaign_link": "https://shop.example/aff",
                "batch_subjects": "SP1\nSP2",
                "product_ideas": [{"product": "idea1"}],
                "content_calendar": [{"subject": "day1"}],
                "ui_language": "vi",
            }
        )
        return state

    def test_text_keys_are_blanked(self):
        state = self._full_canvas()
        apply_new_video_reset(state)
        for key in NEW_VIDEO_TEXT_KEYS:
            self.assertEqual(state[key], "", f"{key} not blanked")

    def test_asset_keys_are_removed(self):
        state = self._full_canvas()
        apply_new_video_reset(state)
        for key in NEW_VIDEO_ASSET_KEYS:
            self.assertNotIn(key, state, f"{key} not removed")

    def test_account_and_planning_state_survives(self):
        state = self._full_canvas()
        apply_new_video_reset(state)
        self.assertEqual(state["channel_niche_input"], "đồ gia dụng nhà bếp")
        self.assertEqual(state["campaign_shop"], "MyShop")
        self.assertEqual(state["campaign_link"], "https://shop.example/aff")
        self.assertEqual(state["batch_subjects"], "SP1\nSP2")
        self.assertEqual(state["product_ideas"], [{"product": "idea1"}])
        self.assertEqual(state["content_calendar"], [{"subject": "day1"}])
        self.assertEqual(state["ui_language"], "vi")

    def test_preserved_keys_never_overlap_cleared_keys(self):
        # A key can't be both preserved and cleared — guards against a future
        # edit accidentally adding an account-level key to a clear list.
        cleared = set(NEW_VIDEO_TEXT_KEYS) | set(NEW_VIDEO_ASSET_KEYS)
        self.assertEqual(cleared & set(NEW_VIDEO_PRESERVED_KEYS), set())

    def test_missing_asset_keys_do_not_raise(self):
        # A canvas that never generated any assets still resets cleanly.
        state = {"video_subject": "x"}
        apply_new_video_reset(state)
        self.assertEqual(state["video_subject"], "")


if __name__ == "__main__":
    unittest.main()
