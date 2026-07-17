"""Pure image helpers for the WebUI subtitle preview.

Extracted from webui/Main.py so the PIL geometry (scaling, padding, background
box, stroke) is unit-testable without Streamlit. The WebUI passes a full font
path; everything here is deterministic given its arguments.
"""

from loguru import logger
from PIL import Image, ImageDraw, ImageFont


def hex_to_rgb(color):
    """Parse a #RRGGBB string to an (r, g, b) tuple, falling back to black for
    malformed values so the preview never raises mid-render."""
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        try:
            return (
                int(color[1:3], 16),
                int(color[3:5], 16),
                int(color[5:7], 16),
            )
        except ValueError:
            pass
    return (0, 0, 0)


def render_subtitle_preview(
    font_path,
    font_size,
    text_fore_color,
    stroke_color,
    stroke_width,
    background_color,
    rounded_background,
    sample_text,
):
    """Render a still image that mimics how a subtitle line will look in the
    final video, so the user can confirm the font, colours, stroke and
    background (and especially Vietnamese diacritics) before generating.

    The geometry mirrors app/services/video.create_text_clip: a 1080px-wide
    reference frame is assumed and font/stroke are scaled down to the preview
    canvas, giving a faithful relative size. background_color of None / False
    means no subtitle background.
    """
    canvas_w = 720
    # Subtitle font sizes in the pipeline are absolute pixels on a ~1080px-wide
    # frame; scale them to the preview canvas to preserve relative proportions.
    scale = canvas_w / 1080.0
    pv_font_size = max(14, int(round(font_size * scale)))
    pv_stroke = max(0, int(round((stroke_width or 0) * scale)))
    spacing = int(pv_font_size * 0.25)

    try:
        font = ImageFont.truetype(font_path, pv_font_size)
    except Exception as exc:  # missing/corrupt font file
        logger.warning(f"subtitle preview: failed to load font {font_path}: {exc}")
        font = ImageFont.load_default()

    measure = ImageDraw.Draw(Image.new("RGB", (canvas_w, 10)))
    bbox = measure.multiline_textbbox(
        (0, 0),
        sample_text,
        font=font,
        align="center",
        spacing=spacing,
        stroke_width=pv_stroke,
    )
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x = int(pv_font_size * 0.6)
    pad_y = int(pv_font_size * 0.4)
    canvas_h = text_h + pad_y * 2 + int(pv_font_size * 1.4)

    # Mid-grey stand-in for a video frame so both light and dark subtitle
    # colours stay visible; a subtle checker hints "this is a placeholder".
    img = Image.new("RGB", (canvas_w, canvas_h), (64, 64, 64))
    draw = ImageDraw.Draw(img, "RGBA")
    tile = 36
    for ty in range(0, canvas_h, tile):
        for tx in range(0, canvas_w, tile):
            if (tx // tile + ty // tile) % 2 == 0:
                draw.rectangle([tx, ty, tx + tile, ty + tile], fill=(74, 74, 74))

    cx, cy = canvas_w // 2, canvas_h // 2
    has_bg = bool(background_color) and isinstance(background_color, str)
    if has_bg:
        box_w = text_w + pad_x * 2
        box_h = text_h + pad_y * 2
        x0, y0 = cx - box_w // 2, cy - box_h // 2
        rgb = hex_to_rgb(background_color)
        radius = int(pv_font_size * 0.4) if rounded_background else 0
        draw.rounded_rectangle(
            [x0, y0, x0 + box_w, y0 + box_h],
            radius=radius,
            fill=(rgb[0], rgb[1], rgb[2], 140),
        )

    draw.multiline_text(
        (cx, cy),
        sample_text,
        font=font,
        fill=text_fore_color,
        anchor="mm",
        align="center",
        spacing=spacing,
        stroke_width=pv_stroke,
        stroke_fill=stroke_color,
    )
    return img
