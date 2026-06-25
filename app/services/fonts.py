"""Subtitle font selection helpers with no UI dependency.

Kept out of ``webui/Main.py`` so the language→font logic (notably the Vietnamese
diacritics handling) can be unit-tested without importing Streamlit.
"""

# Fonts that correctly render a given language's glyphs/diacritics.
# Generic CJK fonts (e.g. MicrosoftYaHei) do not render Vietnamese tone
# marks reliably, so prefer a Vietnamese-designed font for vi content.
RECOMMENDED_FONTS_BY_LANG = {
    "vi": ["BeVietnamPro-Bold.ttf", "UTM Kabel KT.ttf"],
    "zh": ["MicrosoftYaHeiBold.ttc", "STHeitiMedium.ttc"],
}

_DEFAULT_FONT = "MicrosoftYaHeiBold.ttc"


def get_recommended_font(language_code, available_fonts):
    """Return the best-fit subtitle font for a language, falling back to the
    first available font when no preferred font is installed.

    language_code may be a UI code ("vi") or a locale ("vi-VN"); only the
    leading language part is used for matching.
    """
    lang = (language_code or "").split("-")[0].lower()
    for preferred in RECOMMENDED_FONTS_BY_LANG.get(lang, []):
        if preferred in available_fonts:
            return preferred
    if _DEFAULT_FONT in available_fonts:
        return _DEFAULT_FONT
    return available_fonts[0] if available_fonts else _DEFAULT_FONT
