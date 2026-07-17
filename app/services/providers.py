"""Provider-catalog helpers for the WebUI settings panel.

Extracted from webui/Main.py so the HTTP + normalization logic is testable
without Streamlit; the WebUI wraps these in st.cache_data at the call site.
"""

import requests
from loguru import logger

GROQ_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


def fetch_groq_model_ids(api_key: str, base_url: str = "") -> list[str]:
    """List the model ids available to a Groq API key, sorted and deduped.

    Returns [] for a missing key or on any request/shape failure so the
    settings panel can quietly fall back to a manual model-name box.
    """
    if not api_key:
        return []

    normalized_base_url = (base_url or GROQ_DEFAULT_BASE_URL).strip().rstrip("/")
    models_url = f"{normalized_base_url}/models"

    try:
        response = requests.get(
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])

        model_ids = []
        for item in data:
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    model_ids.append(model_id.strip())

        return sorted(set(model_ids))
    except Exception as e:
        logger.warning(f"failed to fetch groq models: {e}")
        return []
