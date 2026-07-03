"""Readiness checks that gate the AI generators, with no UI dependency.

Kept out of ``webui/Main.py`` so the "is the app ready to make an LLM call?"
logic can be unit-tested without importing Streamlit. The WebUI uses this to
warn a non-technical user *before* they click a "Generate…" button and hit a
raw LLM error — the single most common first-run stumbling block.
"""

# Providers that run locally or on a free public endpoint and therefore do not
# require the user to paste an API key before the first request. Everything else
# needs a ``{provider}_api_key`` configured.
LLM_PROVIDERS_WITHOUT_KEY = {"ollama", "g4f", "pollinations"}

# No real LLM API key is this short (Gemini/OpenAI/Moonshot keys run 30+ chars).
# Anything shorter is a placeholder or a truncated paste — treating it as "ready"
# just moves the failure to a raw 401 deep inside a render. Found in the wild:
# a config.toml carrying `gemini_api_key = "AQ.k"` after a save-while-running
# clobbered the file.
MIN_PLAUSIBLE_KEY_LENGTH = 8


def llm_provider_requires_key(provider):
    """Return True when the given LLM provider needs an API key to be usable."""
    return (provider or "").strip().lower() not in LLM_PROVIDERS_WITHOUT_KEY


def is_llm_ready(provider, api_key):
    """Return True when the selected LLM provider can make a call right now.

    Key-less providers (Ollama, G4f, Pollinations) are always ready; every other
    provider is ready only once a plausibly complete API key has been entered —
    present AND long enough to possibly be real, so a truncated key warns up
    front instead of 401-ing mid-render.
    """
    if not llm_provider_requires_key(provider):
        return True
    return len((api_key or "").strip()) >= MIN_PLAUSIBLE_KEY_LENGTH
