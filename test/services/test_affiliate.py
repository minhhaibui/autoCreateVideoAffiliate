import unittest

from app.services.affiliate import build_affiliate_package_text


def _build(**kwargs):
    """Call the bundler with identity labels (so section headings are the raw
    keys) and sensible empty defaults, overriding only what a test cares about."""
    params = dict(
        subject="",
        script="",
        keywords="",
        hooks=[],
        social_meta=None,
        label=lambda key: key,
    )
    params.update(kwargs)
    return build_affiliate_package_text(**params)


class TestBuildAffiliatePackageText(unittest.TestCase):
    def test_all_empty_returns_blank(self):
        self.assertEqual(_build().strip(), "")

    def test_subject_and_script_sections(self):
        out = _build(subject="Mini blender", script="Hook then demo.")
        self.assertIn("## subject\nMini blender", out)
        self.assertIn("## script\nHook then demo.", out)

    def test_empty_sections_are_skipped(self):
        out = _build(subject="Only subject")
        self.assertNotIn("## script", out)
        self.assertNotIn("## keywords", out)

    def test_hooks_are_numbered(self):
        out = _build(hooks=["First hook", "Second hook"])
        self.assertIn("## hooks", out)
        self.assertIn("1. First hook", out)
        self.assertIn("2. Second hook", out)

    def test_shots_include_subfields(self):
        shots = [
            {
                "scene": "Open box",
                "voiceover": "Look at this",
                "onscreen_text": "NEW",
                "broll": "unboxing",
            }
        ]
        out = _build(shots=shots)
        self.assertIn("1. Open box", out)
        self.assertIn("shot_voiceover: Look at this", out)
        self.assertIn("shot_onscreen: NEW", out)
        self.assertIn("shot_broll: unboxing", out)

    def test_shots_omit_missing_subfields(self):
        out = _build(shots=[{"scene": "Just a scene"}])
        self.assertIn("1. Just a scene", out)
        self.assertNotIn("shot_voiceover", out)

    def test_social_meta_sections(self):
        meta = {
            "title": "My title",
            "caption": "My caption",
            "hashtags": ["#a", "#b"],
        }
        out = _build(social_meta=meta)
        self.assertIn("## title\nMy title", out)
        self.assertIn("## caption\nMy caption", out)
        self.assertIn("## hashtags\n#a #b", out)

    def test_comment_replies(self):
        replies = [{"comment": "How much?", "reply": "Link in bio"}]
        out = _build(comment_replies=replies)
        self.assertIn("1. How much?", out)
        self.assertIn("reply: Link in bio", out)

    def test_sound_ideas(self):
        sounds = [{"sound": "Upbeat pop", "vibe": "energetic", "search": "pop"}]
        out = _build(sound_ideas=sounds)
        self.assertIn("1. Upbeat pop", out)
        self.assertIn("sound_vibe: energetic", out)
        self.assertIn("sound_search: pop", out)

    def test_text_stickers(self):
        stickers = [{"text": "Wait for it", "timing": "0-2s", "purpose": "hook"}]
        out = _build(stickers=stickers)
        self.assertIn("1. Wait for it", out)
        self.assertIn("sticker_timing: 0-2s", out)
        self.assertIn("sticker_purpose: hook", out)

    def test_cover_ideas(self):
        covers = [{"text": "Under $20?!", "angle": "benefit", "tip": "bright frame"}]
        out = _build(cover_ideas=covers)
        self.assertIn("1. Under $20?!", out)
        self.assertIn("cover_angle: benefit", out)
        self.assertIn("cover_tip: bright frame", out)

    def test_schedule_slots(self):
        slots = [
            {
                "slot": "Prime evening",
                "time": "7:00-9:00 PM",
                "day": "Weekdays",
                "why": "after-work scroll",
            }
        ]
        out = _build(schedule_slots=slots)
        self.assertIn("1. Prime evening — 7:00-9:00 PM", out)
        self.assertIn("schedule_day: Weekdays", out)
        self.assertIn("schedule_why: after-work scroll", out)

    def test_label_callable_translates_headings(self):
        out = build_affiliate_package_text(
            subject="X",
            script="",
            keywords="",
            hooks=[],
            social_meta=None,
            label=lambda key: {"subject": "Chủ đề"}.get(key, key),
        )
        self.assertIn("## Chủ đề\nX", out)


if __name__ == "__main__":
    unittest.main()
