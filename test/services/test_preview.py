import os
import unittest

from PIL import Image

from app.services import preview
from app.utils import utils

FONT_DIR = os.path.join(utils.root_dir(), "resource", "fonts")
VI_FONT = os.path.join(FONT_DIR, "BeVietnamPro-Bold.ttf")


class TestHexToRgb(unittest.TestCase):
    def test_valid_hex_parses(self):
        self.assertEqual(preview.hex_to_rgb("#FFC0CB"), (255, 192, 203))

    def test_black_and_white(self):
        self.assertEqual(preview.hex_to_rgb("#000000"), (0, 0, 0))
        self.assertEqual(preview.hex_to_rgb("#ffffff"), (255, 255, 255))

    def test_short_hex_falls_back_to_black(self):
        self.assertEqual(preview.hex_to_rgb("#FFF"), (0, 0, 0))

    def test_named_color_falls_back_to_black(self):
        self.assertEqual(preview.hex_to_rgb("red"), (0, 0, 0))

    def test_non_hex_digits_fall_back_to_black(self):
        self.assertEqual(preview.hex_to_rgb("#GGHHII"), (0, 0, 0))

    def test_none_and_empty_fall_back_to_black(self):
        self.assertEqual(preview.hex_to_rgb(None), (0, 0, 0))
        self.assertEqual(preview.hex_to_rgb(""), (0, 0, 0))

    def test_missing_hash_falls_back_to_black(self):
        self.assertEqual(preview.hex_to_rgb("FFC0CB"), (0, 0, 0))


class TestRenderSubtitlePreview(unittest.TestCase):
    def render(self, **overrides):
        kwargs = dict(
            font_path=VI_FONT,
            font_size=60,
            text_fore_color="#FFFFFF",
            stroke_color="#000000",
            stroke_width=2,
            background_color=None,
            rounded_background=False,
            sample_text="Tiếng Việt có dấu — 0123\nSiêu sale hôm nay!",
        )
        kwargs.update(overrides)
        return preview.render_subtitle_preview(**kwargs)

    def test_returns_pil_image_with_preview_width(self):
        img = self.render()
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.width, 720)
        self.assertGreater(img.height, 0)

    def test_missing_font_falls_back_without_raising(self):
        img = self.render(font_path=os.path.join(FONT_DIR, "no-such-font.ttf"))
        self.assertIsInstance(img, Image.Image)

    def test_empty_font_path_falls_back_without_raising(self):
        img = self.render(font_path="")
        self.assertIsInstance(img, Image.Image)

    def test_background_box_darkens_centre_row_edges(self):
        # The translucent background box only covers the text area; sample a
        # pixel inside the box (just off-centre so glyphs are unlikely) on both
        # renders and require the bg render to differ from the no-bg render.
        no_bg = self.render(sample_text="Ab", background_color=None)
        with_bg = self.render(sample_text="Ab", background_color="#FF0000")
        x, y = no_bg.width // 2, no_bg.height // 2
        # Scan the centre row: at least one pixel must gain red from the box.
        gained_red = any(
            with_bg.getpixel((px, y))[0] > no_bg.getpixel((px, y))[0] + 20
            for px in range(x - 30, x + 30)
        )
        self.assertTrue(gained_red)

    def test_rounded_background_renders(self):
        img = self.render(background_color="#000000", rounded_background=True)
        self.assertIsInstance(img, Image.Image)

    def test_none_stroke_width_is_treated_as_zero(self):
        img = self.render(stroke_width=None)
        self.assertIsInstance(img, Image.Image)

    def test_tiny_font_size_clamped_to_legible_floor(self):
        # font_size=1 scales below the 14px floor; must still render.
        img = self.render(font_size=1)
        self.assertIsInstance(img, Image.Image)

    def test_larger_font_size_grows_canvas_height(self):
        small = self.render(font_size=40, sample_text="Ab")
        large = self.render(font_size=120, sample_text="Ab")
        self.assertGreater(large.height, small.height)

    def test_single_line_shorter_than_two_lines(self):
        one = self.render(sample_text="Một dòng")
        two = self.render(sample_text="Dòng một\nDòng hai")
        self.assertGreater(two.height, one.height)


if __name__ == "__main__":
    unittest.main()
