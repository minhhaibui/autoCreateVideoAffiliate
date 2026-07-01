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


def llm_provider_requires_key(provider):
    """Return True when the given LLM provider needs an API key to be usable."""
    return (provider or "").strip().lower() not in LLM_PROVIDERS_WITHOUT_KEY


def is_llm_ready(provider, api_key):
    """Return True when the selected LLM provider can make a call right now.

    Key-less providers (Ollama, G4f, Pollinations) are always ready; every other
    provider is ready only once a non-empty API key has been entered.
    """
    if not llm_provider_requires_key(provider):
        return True
    return bool((api_key or "").strip())
