import unittest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import task as tm
from app.models.schema import MaterialInfo, VideoParams

resources_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")

class TestTaskService(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_generate_script_forwards_advanced_prompt_options(self):
        """
        任务生成入口和 WebUI/API 共用 VideoParams。这里验证自动生成文案时，
        高级提示词参数会继续传到 LLM 服务层，避免只在 /scripts 接口生效。
        """
        params = VideoParams(
            video_subject="咖啡",
            video_script="",
            video_language="zh-CN",
            paragraph_number=2,
            video_script_prompt="语气轻松",
            custom_system_prompt="Only write short narration.",
        )

        with patch.object(tm.llm, "generate_script", return_value="生成的文案") as generate:
            result = tm.generate_script("task-id", params)

        self.assertEqual(result, "生成的文案")
        generate.assert_called_once_with(
            video_subject="咖啡",
            language="zh-CN",
            paragraph_number=2,
            video_script_prompt="语气轻松",
            custom_system_prompt="Only write short narration.",
        )
    
    def test_generate_terms_llm_error_string_fails_the_task_cleanly(self):
        """Nếu tầng LLM vẫn trả về thứ gì đó không phải list (vd chuỗi lỗi
        429), task phải thất bại sạch (None + state FAILED) thay vì đưa chuỗi
        truthy đó xuống bước tải liệu."""
        params = VideoParams(
            video_subject="Máy xay sinh tố", video_script="", video_terms=""
        )
        with patch.object(
            tm.llm, "generate_terms", return_value="Error: 429 quota exceeded"
        ), patch.object(tm.sm.state, "update_task") as update:
            result = tm.generate_terms("task-id", params, "script text")
        self.assertIsNone(result)
        update.assert_called_once()

    def test_task_local_materials(self):
        task_id = "00000000-0000-0000-0000-000000000000"
        video_materials=[]
        for i in range(1, 4):
            video_materials.append(MaterialInfo(
                provider="local",
                url=os.path.join(resources_dir, f"{i}.png"),
                duration=0
            ))

        params = VideoParams(
            video_subject="金钱的作用",
            video_script="金钱不仅是交换媒介，更是社会资源的分配工具。它能满足基本生存需求，如食物和住房，也能提供教育、医疗等提升生活品质的机会。拥有足够的金钱意味着更多选择权，比如职业自由或创业可能。但金钱的作用也有边界，它无法直接购买幸福、健康或真诚的人际关系。过度追逐财富可能导致价值观扭曲，忽视精神层面的需求。理想的状态是理性看待金钱，将其作为实现目标的工具而非终极目的。",
            video_terms="money importance, wealth and society, financial freedom, money and happiness, role of money",
            video_aspect="9:16",
            video_concat_mode="random",
            video_transition_mode="None",
            video_clip_duration=3,
            video_count=1,
            video_source="local",
            video_materials=video_materials,
            video_language="",
            voice_name="zh-CN-XiaoxiaoNeural-Female",
            voice_volume=1.0,
            voice_rate=1.0,
            bgm_type="random",
            bgm_file="",
            bgm_volume=0.2,
            subtitle_enabled=True,
            subtitle_position="bottom",
            custom_position=70.0,
            font_name="MicrosoftYaHeiBold.ttc",
            text_fore_color="#FFFFFF",
            text_background_color=True,
            font_size=60,
            stroke_color="#000000",
            stroke_width=1.5,
            n_threads=2,
            paragraph_number=1
        )
        result = tm.start(task_id=task_id, params=params)
        print(result)


class TestProductMaterials(unittest.TestCase):
    """Real product media must open the video, weave into stock, and never
    make a render fail that would otherwise succeed."""

    def test_no_product_materials_returns_empty(self):
        params = VideoParams(video_subject="x")
        self.assertEqual(tm.preprocess_product_materials(params), [])

    def test_preprocess_returns_video_ready_urls_from_product_dir(self):
        params = VideoParams(
            video_subject="x",
            product_materials=[MaterialInfo(provider="local", url="p.jpg")],
        )
        processed = [MaterialInfo(provider="local", url="p.jpg.mp4")]
        with patch.object(
            tm.video, "preprocess_video", return_value=processed
        ) as mock_pre:
            self.assertEqual(tm.preprocess_product_materials(params), ["p.jpg.mp4"])
        self.assertTrue(
            mock_pre.call_args.kwargs["materials_dir"].endswith("product_media")
        )

    def test_preprocess_failure_never_fails_the_render(self):
        params = VideoParams(
            video_subject="x",
            product_materials=[MaterialInfo(provider="local", url="p.jpg")],
        )
        with patch.object(
            tm.video, "preprocess_video", side_effect=RuntimeError("boom")
        ):
            self.assertEqual(tm.preprocess_product_materials(params), [])

    def test_online_source_weaves_product_clip_first(self):
        params = VideoParams(
            video_subject="x",
            video_source="pexels",
            product_materials=[MaterialInfo(provider="local", url="p.jpg")],
        )
        processed = [MaterialInfo(provider="local", url="p.mp4")]
        with patch.object(tm.video, "preprocess_video", return_value=processed), patch.object(
            tm.material, "download_videos", return_value=["s1.mp4", "s2.mp4"]
        ):
            result = tm.get_video_materials("tid", params, ["term"], 30)
        self.assertEqual(result, ["p.mp4", "s1.mp4", "s2.mp4"])

    def test_stock_download_failure_with_product_media_still_renders(self):
        params = VideoParams(
            video_subject="x",
            video_source="pexels",
            product_materials=[MaterialInfo(provider="local", url="p.jpg")],
        )
        processed = [MaterialInfo(provider="local", url="p.mp4")]
        with patch.object(tm.video, "preprocess_video", return_value=processed), patch.object(
            tm.material, "download_videos", return_value=[]
        ), patch.object(tm.sm.state, "update_task") as mock_state:
            result = tm.get_video_materials("tid", params, ["term"], 30)
        self.assertEqual(result, ["p.mp4"])
        mock_state.assert_not_called()

    def test_stock_download_failure_without_product_media_fails_task(self):
        params = VideoParams(video_subject="x", video_source="pexels")
        with patch.object(
            tm.material, "download_videos", return_value=[]
        ), patch.object(tm.sm.state, "update_task") as mock_state:
            result = tm.get_video_materials("tid", params, ["term"], 30)
        self.assertIsNone(result)
        mock_state.assert_called_once()

    def test_generate_final_videos_pins_product_urls(self):
        params = VideoParams(
            video_subject="x",
            video_count=1,
            product_materials=[
                MaterialInfo(provider="local", url="p.mp4"),
                MaterialInfo(provider="local", url=""),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(tm.utils, "task_dir", return_value=tmp), patch.object(
                tm.video, "combine_videos"
            ) as mock_combine, patch.object(tm.video, "generate_video"), patch.object(
                tm.sm.state, "update_task"
            ):
                tm.generate_final_videos(
                    "tid", params, ["p.mp4", "s1.mp4"], "a.mp3", "s.srt"
                )
        self.assertEqual(mock_combine.call_args.kwargs["pinned_paths"], ["p.mp4"])

    def test_generate_final_videos_without_product_pins_nothing(self):
        params = VideoParams(video_subject="x", video_count=1)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(tm.utils, "task_dir", return_value=tmp), patch.object(
                tm.video, "combine_videos"
            ) as mock_combine, patch.object(tm.video, "generate_video"), patch.object(
                tm.sm.state, "update_task"
            ):
                tm.generate_final_videos("tid", params, ["s1.mp4"], "a.mp3", "s.srt")
        self.assertEqual(mock_combine.call_args.kwargs["pinned_paths"], [])


if __name__ == "__main__":
    unittest.main()
