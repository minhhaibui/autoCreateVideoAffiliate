import unittest

from app.services.fonts import get_recommended_font


FONTS = [
    "BeVietnamPro-Bold.ttf",
    "Charm-Bold.ttf",
    "MicrosoftYaHeiBold.ttc",
    "STHeitiMedium.ttc",
    "UTM Kabel KT.ttf",
]


class TestGetRecommendedFont(unittest.TestCase):
    def test_vietnamese_prefers_be_vietnam_pro(self):
        self.assertEqual(get_recommended_font("vi", FONTS), "BeVietnamPro-Bold.ttf")

    def test_locale_code_is_reduced_to_language(self):
        self.assertEqual(get_recommended_font("vi-VN", FONTS), "BeVietnamPro-Bold.ttf")

    def test_language_code_is_case_insensitive(self):
        self.assertEqual(get_recommended_font("VI", FONTS), "BeVietnamPro-Bold.ttf")

    def test_vietnamese_falls_back_to_second_choice(self):
        # BeVietnamPro missing -> next preferred Vietnamese font is used,
        # NOT a generic CJK font that mangles tone marks.
        fonts = [f for f in FONTS if f != "BeVietnamPro-Bold.ttf"]
        self.assertEqual(get_recommended_font("vi", fonts), "UTM Kabel KT.ttf")

    def test_chinese_prefers_microsoft_yahei(self):
        self.assertEqual(get_recommended_font("zh", FONTS), "MicrosoftYaHeiBold.ttc")

    def test_unknown_language_falls_back_to_yahei_when_present(self):
        self.assertEqual(get_recommended_font("en", FONTS), "MicrosoftYaHeiBold.ttc")

    def test_unknown_language_without_yahei_uses_first_available(self):
        fonts = ["Charm-Bold.ttf", "STHeitiMedium.ttc"]
        self.assertEqual(get_recommended_font("en", fonts), "Charm-Bold.ttf")

    def test_empty_font_list_returns_default_name(self):
        self.assertEqual(get_recommended_font("vi", []), "MicrosoftYaHeiBold.ttc")

    def test_none_language_does_not_crash(self):
        self.assertEqual(get_recommended_font(None, FONTS), "MicrosoftYaHeiBold.ttc")


if __name__ == "__main__":
    unittest.main()
