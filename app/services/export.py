"""TikTok-ready export: keep every final video under the upload size cap.

The renderer writes ~12 Mbps files (a 36s video ≈ 66MB); TikTok web upload
(and the Chrome file_upload flow) caps at 10MB, which used to mean a manual
ffmpeg re-encode before every post. This module computes a bitrate that fits
the cap and produces `<final>-tiktok.mp4` automatically after each render.

Pure math is separated from the ffmpeg invocation so the sizing logic is
unit-testable without encoding anything.
"""

import os
import subprocess

from loguru import logger

from app.utils import utils

TIKTOK_SIZE_LIMIT_BYTES = 10 * 1024 * 1024
AUDIO_KBPS = 128
# Leave headroom for container overhead and bitrate-control variance.
SAFETY = 0.90
MIN_VIDEO_KBPS = 300


def tiktok_video_kbps(
    duration_seconds: float,
    size_limit_bytes: int = TIKTOK_SIZE_LIMIT_BYTES,
    audio_kbps: int = AUDIO_KBPS,
) -> int:
    """Video bitrate (kbps) that fits ``size_limit_bytes`` for this duration,
    after reserving the audio track and a safety margin. Floors at
    MIN_VIDEO_KBPS so absurd durations still produce a playable file (the
    post-encode size check is the real gate)."""
    if duration_seconds <= 0:
        return MIN_VIDEO_KBPS
    total_kbps = (size_limit_bytes * 8 / 1000.0) / duration_seconds * SAFETY
    return max(MIN_VIDEO_KBPS, int(total_kbps - audio_kbps))


def _run_ffmpeg_encode(video_path: str, output_path: str, video_kbps: int) -> None:
    cmd = [
        utils.get_ffmpeg_binary(),
        "-y",
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-b:v",
        f"{video_kbps}k",
        "-maxrate",
        f"{int(video_kbps * 1.2)}k",
        "-bufsize",
        f"{video_kbps * 2}k",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        f"{AUDIO_KBPS}k",
        "-movflags",
        "+faststart",
        output_path,
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def ensure_tiktok_ready(
    video_path: str,
    output_path: str,
    duration_seconds: float,
    size_limit_bytes: int = TIKTOK_SIZE_LIMIT_BYTES,
) -> str:
    """Return a path whose file fits the TikTok upload cap.

    Already small enough → the original path is returned and nothing is
    written. Otherwise re-encode to ``output_path`` at a computed bitrate,
    retrying once at 0.75x if the first pass still lands over the cap.
    Raises on encode failure — the caller decides whether that is fatal
    (task.start treats it as never-fatal: the full-size final still exists).
    """
    source_size = os.path.getsize(video_path)
    if source_size <= size_limit_bytes:
        logger.info(
            f"tiktok-ready: {os.path.basename(video_path)} already fits "
            f"({source_size / 1024 / 1024:.1f}MB <= {size_limit_bytes / 1024 / 1024:.0f}MB)"
        )
        return video_path

    kbps = tiktok_video_kbps(duration_seconds, size_limit_bytes)
    for attempt_kbps in (kbps, max(MIN_VIDEO_KBPS, int(kbps * 0.75))):
        logger.info(
            f"tiktok-ready: encoding {os.path.basename(output_path)} at {attempt_kbps} kbps "
            f"(source {source_size / 1024 / 1024:.1f}MB, {duration_seconds:.0f}s)"
        )
        _run_ffmpeg_encode(video_path, output_path, attempt_kbps)
        out_size = os.path.getsize(output_path)
        if out_size <= size_limit_bytes:
            logger.success(
                f"tiktok-ready: {os.path.basename(output_path)} = "
                f"{out_size / 1024 / 1024:.1f}MB — ready to upload"
            )
            return output_path
    logger.warning(
        f"tiktok-ready: still {out_size / 1024 / 1024:.1f}MB over the cap after retry; "
        "keeping the smaller file anyway"
    )
    return output_path
