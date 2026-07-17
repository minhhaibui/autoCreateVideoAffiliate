import os
import sys
import tempfile
import unittest
from unittest.mock import patch

root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.services import export


class TestTiktokVideoKbps(unittest.TestCase):
    def test_36s_video_fits_10mb(self):
        kbps = export.tiktok_video_kbps(36)
        # total budget must stay under the cap: (video+audio) * duration
        total_bytes = (kbps + export.AUDIO_KBPS) * 1000 / 8 * 36
        self.assertLessEqual(total_bytes, export.TIKTOK_SIZE_LIMIT_BYTES)
        self.assertGreater(kbps, export.MIN_VIDEO_KBPS)

    def test_longer_video_gets_lower_bitrate(self):
        self.assertLess(export.tiktok_video_kbps(120), export.tiktok_video_kbps(30))

    def test_absurd_duration_floors_at_minimum(self):
        self.assertEqual(export.tiktok_video_kbps(100000), export.MIN_VIDEO_KBPS)

    def test_zero_and_negative_duration_floor_at_minimum(self):
        self.assertEqual(export.tiktok_video_kbps(0), export.MIN_VIDEO_KBPS)
        self.assertEqual(export.tiktok_video_kbps(-5), export.MIN_VIDEO_KBPS)


class TestEnsureTiktokReady(unittest.TestCase):
    def test_small_file_returned_unchanged_without_encoding(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"x" * 1024)
        self.addCleanup(os.unlink, f.name)
        with patch.object(export, "_run_ffmpeg_encode") as mock_enc:
            result = export.ensure_tiktok_ready(f.name, f.name + "-tiktok.mp4", 30)
        mock_enc.assert_not_called()
        self.assertEqual(result, f.name)

    def test_large_file_is_reencoded_once_when_it_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "final-1.mp4")
            out = os.path.join(tmp, "final-1-tiktok.mp4")
            with open(src, "wb") as f:
                f.write(b"x" * (export.TIKTOK_SIZE_LIMIT_BYTES + 1))

            def fake_encode(video_path, output_path, kbps):
                with open(output_path, "wb") as f:
                    f.write(b"y" * 1024)

            with patch.object(
                export, "_run_ffmpeg_encode", side_effect=fake_encode
            ) as mock_enc:
                result = export.ensure_tiktok_ready(src, out, 30)
            self.assertEqual(mock_enc.call_count, 1)
            self.assertEqual(result, out)

    def test_still_too_big_retries_once_at_lower_bitrate(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "final-1.mp4")
            out = os.path.join(tmp, "final-1-tiktok.mp4")
            with open(src, "wb") as f:
                f.write(b"x" * (export.TIKTOK_SIZE_LIMIT_BYTES + 1))

            def fat_encode(video_path, output_path, kbps):
                with open(output_path, "wb") as f:
                    f.write(b"y" * (export.TIKTOK_SIZE_LIMIT_BYTES + 1))

            with patch.object(
                export, "_run_ffmpeg_encode", side_effect=fat_encode
            ) as mock_enc:
                result = export.ensure_tiktok_ready(src, out, 30)
            self.assertEqual(mock_enc.call_count, 2)
            first_kbps = mock_enc.call_args_list[0].args[2]
            second_kbps = mock_enc.call_args_list[1].args[2]
            self.assertLess(second_kbps, first_kbps)
            self.assertEqual(result, out)


class TestAutopilotReportReadyVideos(unittest.TestCase):
    def test_report_lists_upload_ready_file_when_different(self):
        from app.services import autopilot

        report = autopilot.format_report(
            {"product": "Máy xay"},
            ["/t/final-1.mp4"],
            {},
            [],
            ready_videos=["/t/final-1-tiktok.mp4"],
        )
        self.assertIn("Upload-ready (<=10MB): /t/final-1-tiktok.mp4", report)

    def test_report_skips_ready_line_when_final_already_fits(self):
        from app.services import autopilot

        report = autopilot.format_report(
            {"product": "Máy xay"},
            ["/t/final-1.mp4"],
            {},
            [],
            ready_videos=["/t/final-1.mp4"],
        )
        self.assertNotIn("Upload-ready", report)


if __name__ == "__main__":
    unittest.main()
