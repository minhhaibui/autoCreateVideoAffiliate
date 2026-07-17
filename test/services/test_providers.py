import unittest
from unittest.mock import MagicMock, patch

import requests

from app.services import providers


def _response(payload):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


class TestFetchGroqModelIds(unittest.TestCase):
    def test_empty_key_returns_empty_without_fetching(self):
        with patch.object(providers.requests, "get") as mock_get:
            self.assertEqual(providers.fetch_groq_model_ids(""), [])
        mock_get.assert_not_called()

    def test_ids_are_deduped_stripped_and_sorted(self):
        payload = {
            "data": [
                {"id": " llama-3.3-70b "},
                {"id": "gemma2-9b"},
                {"id": "llama-3.3-70b"},
            ]
        }
        with patch.object(providers.requests, "get", return_value=_response(payload)):
            ids = providers.fetch_groq_model_ids("key")
        self.assertEqual(ids, ["gemma2-9b", "llama-3.3-70b"])

    def test_malformed_items_are_skipped(self):
        payload = {
            "data": [
                "not-a-dict",
                {"id": 42},
                {"id": "   "},
                {"no_id": "x"},
                {"id": "valid-model"},
            ]
        }
        with patch.object(providers.requests, "get", return_value=_response(payload)):
            ids = providers.fetch_groq_model_ids("key")
        self.assertEqual(ids, ["valid-model"])

    def test_missing_data_key_returns_empty(self):
        with patch.object(providers.requests, "get", return_value=_response({})):
            self.assertEqual(providers.fetch_groq_model_ids("key"), [])

    def test_http_error_returns_empty(self):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError("401")
        with patch.object(providers.requests, "get", return_value=resp):
            self.assertEqual(providers.fetch_groq_model_ids("bad-key"), [])

    def test_network_error_returns_empty(self):
        with patch.object(
            providers.requests, "get", side_effect=requests.ConnectionError("down")
        ):
            self.assertEqual(providers.fetch_groq_model_ids("key"), [])

    def test_default_base_url_is_used(self):
        with patch.object(
            providers.requests, "get", return_value=_response({"data": []})
        ) as mock_get:
            providers.fetch_groq_model_ids("key")
        self.assertEqual(
            mock_get.call_args.args[0], "https://api.groq.com/openai/v1/models"
        )

    def test_custom_base_url_trailing_slash_normalized(self):
        with patch.object(
            providers.requests, "get", return_value=_response({"data": []})
        ) as mock_get:
            providers.fetch_groq_model_ids("key", " https://proxy.example/v1/ ")
        self.assertEqual(mock_get.call_args.args[0], "https://proxy.example/v1/models")

    def test_auth_header_carries_key(self):
        with patch.object(
            providers.requests, "get", return_value=_response({"data": []})
        ) as mock_get:
            providers.fetch_groq_model_ids("sk-test")
        self.assertEqual(
            mock_get.call_args.kwargs["headers"], {"Authorization": "Bearer sk-test"}
        )


if __name__ == "__main__":
    unittest.main()
