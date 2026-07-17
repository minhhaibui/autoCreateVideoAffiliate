import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.config import config
from app.services import upload_post


def make_service(**overrides):
    """Build an UploadPostService against a temporary in-memory config view.

    patch.dict restores config.app afterwards and nothing here calls
    config.save_config(), so the real config.toml is never touched.
    """
    values = {
        "upload_post_api_key": "key-123",
        "upload_post_username": "creator",
        "upload_post_enabled": True,
        "upload_post_platforms": ["tiktok", "instagram"],
        "upload_post_auto_upload": False,
    }
    values.update(overrides)
    with patch.dict(config.app, values):
        return upload_post.UploadPostService()


def json_response(payload):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


class TestIsConfigured(unittest.TestCase):
    def test_fully_configured(self):
        self.assertTrue(make_service().is_configured())

    def test_missing_api_key(self):
        self.assertFalse(make_service(upload_post_api_key="").is_configured())

    def test_missing_username(self):
        self.assertFalse(make_service(upload_post_username="").is_configured())

    def test_disabled(self):
        self.assertFalse(make_service(upload_post_enabled=False).is_configured())


class TestUploadVideo(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        handle.write(b"fake video bytes")
        handle.close()
        self.video_path = handle.name
        self.addCleanup(os.unlink, self.video_path)

    def test_unconfigured_service_skips_without_network(self):
        service = make_service(upload_post_enabled=False)
        with patch.object(upload_post.requests, "post") as mock_post:
            result = service.upload_video(self.video_path, "title")
        mock_post.assert_not_called()
        self.assertFalse(result["success"])
        self.assertIn("not configured", result["error"])

    def test_missing_file_fails_without_network(self):
        service = make_service()
        with patch.object(upload_post.requests, "post") as mock_post:
            result = service.upload_video("/no/such/video.mp4", "title")
        mock_post.assert_not_called()
        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])

    def test_successful_upload_returns_api_payload(self):
        service = make_service()
        payload = {"success": True, "request_id": "req-1"}
        with patch.object(
            upload_post.requests, "post", return_value=json_response(payload)
        ) as mock_post:
            result = service.upload_video(self.video_path, "Nồi chiên không dầu")
        self.assertEqual(result, payload)
        kwargs = mock_post.call_args.kwargs
        self.assertEqual(kwargs["headers"], {"Authorization": "Apikey key-123"})
        self.assertEqual(kwargs["data"]["user"], "creator")
        self.assertEqual(kwargs["data"]["title"], "Nồi chiên không dầu")

    def test_default_platforms_come_from_config(self):
        service = make_service(upload_post_platforms=["tiktok"])
        with patch.object(
            upload_post.requests, "post", return_value=json_response({"success": True})
        ) as mock_post:
            service.upload_video(self.video_path, "t")
        data = mock_post.call_args.kwargs["data"]
        self.assertEqual(data["platform[0]"], "tiktok")
        self.assertNotIn("platform[1]", data)

    def test_explicit_platforms_override_config(self):
        service = make_service()
        with patch.object(
            upload_post.requests, "post", return_value=json_response({"success": True})
        ) as mock_post:
            service.upload_video(self.video_path, "t", platforms=["instagram"])
        data = mock_post.call_args.kwargs["data"]
        self.assertEqual(data["platform[0]"], "instagram")
        self.assertNotIn("platform[1]", data)

    def test_title_is_truncated_to_instagram_limit(self):
        service = make_service()
        with patch.object(
            upload_post.requests, "post", return_value=json_response({"success": True})
        ) as mock_post:
            service.upload_video(self.video_path, "x" * 3000)
        self.assertEqual(len(mock_post.call_args.kwargs["data"]["title"]), 2200)

    def test_network_error_returns_failure_dict(self):
        service = make_service()
        with patch.object(
            upload_post.requests,
            "post",
            side_effect=requests.exceptions.ConnectionError("down"),
        ):
            result = service.upload_video(self.video_path, "t")
        self.assertFalse(result["success"])
        self.assertIn("down", result["error"])

    def test_http_error_returns_failure_dict(self):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("401")
        service = make_service()
        with patch.object(upload_post.requests, "post", return_value=resp):
            result = service.upload_video(self.video_path, "t")
        self.assertFalse(result["success"])
        self.assertIn("401", result["error"])

    def test_api_reported_failure_is_passed_through(self):
        service = make_service()
        payload = {"success": False, "message": "quota exceeded"}
        with patch.object(
            upload_post.requests, "post", return_value=json_response(payload)
        ):
            result = service.upload_video(self.video_path, "t")
        self.assertEqual(result, payload)


class TestCheckStatus(unittest.TestCase):
    def test_status_url_and_auth_header(self):
        service = make_service()
        with patch.object(
            upload_post.requests, "get", return_value=json_response({"status": "done"})
        ) as mock_get:
            result = service.check_status("req-9")
        self.assertEqual(result, {"status": "done"})
        self.assertEqual(
            mock_get.call_args.args[0],
            "https://api.upload-post.com/api/status/req-9",
        )
        self.assertEqual(
            mock_get.call_args.kwargs["headers"], {"Authorization": "Apikey key-123"}
        )

    def test_network_error_returns_failure_dict(self):
        service = make_service()
        with patch.object(
            upload_post.requests,
            "get",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            result = service.check_status("req-9")
        self.assertFalse(result["success"])
        self.assertIn("timed out", result["error"])


class TestCrossPostVideo(unittest.TestCase):
    def test_delegates_to_singleton(self):
        with patch.object(
            upload_post.upload_post_service, "upload_video", return_value={"success": True}
        ) as mock_upload:
            result = upload_post.cross_post_video("/path.mp4", "title", ["tiktok"])
        self.assertEqual(result, {"success": True})
        mock_upload.assert_called_once_with("/path.mp4", "title", ["tiktok"])


if __name__ == "__main__":
    unittest.main()
