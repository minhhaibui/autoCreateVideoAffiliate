import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from app.services import autopilot, llm  # noqa: E402


IDEAS = [
    {"product": "Máy hút bụi mini", "reason": "r1", "angle": "a1",
     "category": "gia dụng", "audience": "x"},
    {"product": "Kệ bếp đa năng", "reason": "r2", "angle": "a2",
     "category": "gia dụng", "audience": "y"},
    {"product": "Bình giữ nhiệt", "reason": "r3", "angle": "a3",
     "category": "gia dụng", "audience": "z"},
]


class TestHistory(unittest.TestCase):
    def test_roundtrip_and_cap(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "history.json")
            self.assertEqual(autopilot.load_history(path), [])
            entries = [{"product": f"p{i}"} for i in range(autopilot.HISTORY_LIMIT + 5)]
            autopilot.save_history(path, entries)
            loaded = autopilot.load_history(path)
            self.assertEqual(len(loaded), autopilot.HISTORY_LIMIT)
            self.assertEqual(loaded[-1]["product"], f"p{autopilot.HISTORY_LIMIT + 4}")

    def test_corrupt_file_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "history.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{not json")
            self.assertEqual(autopilot.load_history(path), [])
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"a": 1}, f)
            self.assertEqual(autopilot.load_history(path), [])


class TestFallbackModel(unittest.TestCase):
    def test_switches_gemini_to_the_fallback_in_process(self):
        app = {"llm_provider": "gemini", "gemini_model_name": "gemini-2.5-flash"}
        with patch.object(autopilot.config, "app", app):
            self.assertEqual(
                autopilot._activate_fallback_model(),
                autopilot.FALLBACK_GEMINI_MODEL,
            )
            self.assertEqual(
                app["gemini_model_name"], autopilot.FALLBACK_GEMINI_MODEL
            )

    def test_configured_fallback_name_wins_over_the_default(self):
        app = {"llm_provider": "gemini", "gemini_model_name": "gemini-2.5-flash",
               "autopilot_fallback_model": "gemini-x"}
        with patch.object(autopilot.config, "app", app):
            self.assertEqual(autopilot._activate_fallback_model(), "gemini-x")
            self.assertEqual(app["gemini_model_name"], "gemini-x")

    def test_non_gemini_provider_is_left_alone(self):
        app = {"llm_provider": "openai", "gemini_model_name": "gemini-2.5-flash"}
        with patch.object(autopilot.config, "app", app):
            self.assertEqual(autopilot._activate_fallback_model(), "")
            self.assertEqual(app["gemini_model_name"], "gemini-2.5-flash")

    def test_already_on_the_fallback_switches_nothing(self):
        app = {"llm_provider": "gemini",
               "gemini_model_name": autopilot.FALLBACK_GEMINI_MODEL}
        with patch.object(autopilot.config, "app", app):
            self.assertEqual(autopilot._activate_fallback_model(), "")


# config.app is patched to a non-Gemini provider so the last-attempt model
# fallback stays inert (and can never mutate the real in-memory config);
# the fallback-specific test re-patches with a Gemini dict of its own.
@patch.object(autopilot, "_cooldown", lambda *a, **k: None)
@patch.object(autopilot.config, "app", {"llm_provider": "none"})
class TestPickBestProduct(unittest.TestCase):
    def test_retries_when_the_pool_comes_back_empty(self):
        """Cạn quota free-tier: lượt đầu rỗng → đợi hết cửa sổ rồi thử lại."""
        with patch.object(
            llm, "generate_product_ideas", side_effect=[[], IDEAS]
        ) as gen, patch.object(llm, "rank_product_ideas", return_value=[]):
            pick = autopilot.pick_best_product("đồ bếp")
        self.assertEqual(pick["product"], "Máy hút bụi mini")
        self.assertEqual(gen.call_count, 2)

    def test_gives_up_after_all_attempts_empty(self):
        with patch.object(
            llm, "generate_product_ideas",
            side_effect=[[]] * autopilot.PICK_ATTEMPTS,
        ) as gen:
            self.assertIsNone(autopilot.pick_best_product("đồ bếp"))
        self.assertEqual(gen.call_count, autopilot.PICK_ATTEMPTS)

    def test_last_attempt_switches_to_the_fallback_model_pool(self):
        """Quota NGÀY của model chính cạn: 2 lượt rỗng → lượt cuối đổi sang
        model dự phòng (pool quota riêng) và pick thành công."""
        app = {"llm_provider": "gemini", "gemini_model_name": "gemini-2.5-flash"}
        with patch.object(autopilot.config, "app", app), \
             patch.object(llm, "generate_product_ideas",
                          side_effect=[[], [], IDEAS]) as gen, \
             patch.object(llm, "rank_product_ideas", return_value=[]):
            pick = autopilot.pick_best_product("đồ bếp")
        self.assertEqual(pick["product"], "Máy hút bụi mini")
        self.assertEqual(gen.call_count, 3)
        self.assertEqual(
            app["gemini_model_name"], autopilot.FALLBACK_GEMINI_MODEL
        )

    def test_success_before_the_last_attempt_keeps_the_primary_model(self):
        app = {"llm_provider": "gemini", "gemini_model_name": "gemini-2.5-flash"}
        with patch.object(autopilot.config, "app", app), \
             patch.object(llm, "generate_product_ideas",
                          side_effect=[[], IDEAS]), \
             patch.object(llm, "rank_product_ideas", return_value=[]):
            pick = autopilot.pick_best_product("đồ bếp")
        self.assertEqual(pick["product"], "Máy hút bụi mini")
        self.assertEqual(app["gemini_model_name"], "gemini-2.5-flash")

    def test_ranked_winner_wins_and_carries_why(self):
        ranked = [
            {"product": "Kệ bếp đa năng", "why": "giá rẻ, nhu cầu rộng"},
            {"product": "Bình giữ nhiệt", "why": "ổn"},
        ]
        with patch.object(llm, "generate_product_ideas", return_value=IDEAS), \
             patch.object(llm, "rank_product_ideas", return_value=ranked):
            pick = autopilot.pick_best_product("đồ bếp")
        self.assertEqual(pick["product"], "Kệ bếp đa năng")
        self.assertEqual(pick["why_best"], "giá rẻ, nhu cầu rộng")
        self.assertEqual(pick["angle"], "a2")

    def test_history_products_are_skipped_case_insensitively(self):
        ranked = [{"product": "Máy hút bụi mini", "why": "best"}]
        history = [{"product": "máy hút bụi mini"}]
        with patch.object(llm, "generate_product_ideas", return_value=IDEAS), \
             patch.object(llm, "rank_product_ideas", return_value=ranked) as rank:
            pick = autopilot.pick_best_product("đồ bếp", history=history)
        # the used product never reaches the ranking pool, so the ranking's
        # echo of it can't match and the fallback is the first FRESH idea
        self.assertEqual(pick["product"], "Kệ bếp đa năng")
        self.assertNotIn(
            "Máy hút bụi mini", [i["product"] for i in rank.call_args[0][0]]
        )

    def test_all_used_allows_repeats_instead_of_failing(self):
        history = [{"product": i["product"]} for i in IDEAS]
        with patch.object(llm, "generate_product_ideas", return_value=IDEAS), \
             patch.object(llm, "rank_product_ideas", return_value=[]):
            pick = autopilot.pick_best_product("đồ bếp", history=history)
        self.assertEqual(pick["product"], "Máy hút bụi mini")
        self.assertEqual(pick["why_best"], "")

    def test_unmatchable_ranking_falls_back_to_pool_order(self):
        ranked = [{"product": "sản phẩm bịa ra", "why": "?"}]
        with patch.object(llm, "generate_product_ideas", return_value=IDEAS), \
             patch.object(llm, "rank_product_ideas", return_value=ranked):
            pick = autopilot.pick_best_product("đồ bếp")
        self.assertEqual(pick["product"], "Máy hút bụi mini")

    def test_no_ideas_returns_none(self):
        with patch.object(llm, "generate_product_ideas", return_value=[]):
            self.assertIsNone(autopilot.pick_best_product("đồ bếp"))

    def test_ideas_missing_product_key_do_not_crash_the_picker(self):
        with patch.object(
            llm, "generate_product_ideas", return_value=[{"reason": "x"}, None]
        ):
            self.assertIsNone(autopilot.pick_best_product("đồ bếp"))


class TestBuildVideoParams(unittest.TestCase):
    def test_params_render_one_portrait_video_from_empty_script(self):
        params = autopilot.build_video_params("Kệ bếp đa năng")
        self.assertEqual(params.video_subject, "Kệ bếp đa năng")
        self.assertEqual(params.video_script, "")
        self.assertEqual(params.video_count, 1)
        self.assertEqual(params.video_source, "pexels")
        self.assertTrue(params.subtitle_enabled)
        self.assertTrue(params.voice_name)


class TestFormatReport(unittest.TestCase):
    def test_full_report(self):
        text = autopilot.format_report(
            {"product": "Kệ bếp", "why_best": "nhu cầu rộng", "angle": "trước/sau"},
            ["/tmp/final-1.mp4"],
            {"title": "T", "caption": "C", "hashtags": ["#a", "#b"]},
            [{"comment": "Ghim: link ở đây", "cta": "mua ngay"}],
        )
        for expected in ["Product: Kệ bếp", "Why picked: nhu cầu rộng",
                         "Video: /tmp/final-1.mp4", "Title: T",
                         "Hashtags: #a #b", "Pinned: Ghim: link ở đây",
                         "real affiliate link"]:
            self.assertIn(expected, text)

    def test_missing_copy_still_reports_video(self):
        text = autopilot.format_report({"product": "Kệ bếp"}, ["/v.mp4"], {}, [])
        self.assertIn("Video: /v.mp4", text)
        self.assertNotIn("Title:", text)


@patch.object(autopilot, "_cooldown", lambda *a, **k: None)
class TestRunAutopilot(unittest.TestCase):
    def _run(self, tmpdir, **overrides):
        """Happy-path harness with every external dependency mocked."""
        task_dir = os.path.join(tmpdir, "task")
        os.makedirs(task_dir, exist_ok=True)
        defaults = dict(
            pick={"product": "Kệ bếp đa năng", "why_best": "w"},
            render={"videos": [os.path.join(task_dir, "final-1.mp4")]},
        )
        defaults.update(overrides)
        with patch.object(autopilot, "history_path",
                          return_value=os.path.join(tmpdir, "history.json")), \
             patch.object(autopilot.utils, "task_dir", return_value=task_dir), \
             patch.object(autopilot, "pick_best_product",
                          return_value=defaults["pick"]), \
             patch.object(autopilot.tm, "start", return_value=defaults["render"]), \
             patch.object(llm, "generate_social_metadata",
                          return_value={"title": "T", "caption": "C",
                                        "hashtags": ["#x"]}), \
             patch.object(llm, "generate_pinned_comments",
                          return_value=[{"comment": "ghim"}]), \
             patch.object(autopilot, "is_llm_ready", return_value=True), \
             patch.object(autopilot.config, "app",
                          {"llm_provider": "gemini", "gemini_api_key": "k" * 20,
                           "pexels_api_keys": ["p"], "channel_niche": "đồ bếp"}):
            return autopilot.run_autopilot()

    def test_happy_path_writes_report_and_history(self):
        with tempfile.TemporaryDirectory() as d:
            result = self._run(d)
            self.assertEqual(result["error"], "")
            self.assertEqual(result["product"], "Kệ bếp đa năng")
            self.assertTrue(result["videos"])
            with open(result["report"], encoding="utf-8") as f:
                self.assertIn("Pinned: ghim", f.read())
            history = autopilot.load_history(os.path.join(d, "history.json"))
            self.assertEqual(history[-1]["product"], "Kệ bếp đa năng")

    def test_render_failure_records_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            result = self._run(d, render=None)
            self.assertIn("render failed", result["error"])
            self.assertEqual(
                autopilot.load_history(os.path.join(d, "history.json")), []
            )

    def test_missing_niche_errors_before_any_llm_call(self):
        with patch.object(autopilot.config, "app", {"channel_niche": ""}):
            result = autopilot.run_autopilot()
        self.assertIn("no niche", result["error"])

    def test_llm_not_ready_errors(self):
        with patch.object(autopilot.config, "app",
                          {"llm_provider": "gemini", "gemini_api_key": "",
                           "channel_niche": "đồ bếp"}):
            result = autopilot.run_autopilot()
        self.assertIn("llm not ready", result["error"])


class TestRankProductIdeas(unittest.TestCase):
    def test_parses_ranked_objects(self):
        payload = json.dumps(
            [{"product": "Kệ bếp đa năng", "why": "giá rẻ"},
             {"product": "Máy hút bụi mini", "why": "ổn"}]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            ranked = llm.rank_product_ideas(IDEAS, niche="đồ bếp")
        self.assertEqual(ranked[0]["product"], "Kệ bếp đa năng")
        for item in ranked:
            self.assertEqual(set(item.keys()), set(llm.RANKING_KEYS))

    def test_prompt_embeds_candidates_and_criteria(self):
        with patch.object(llm, "_generate_response", return_value="[]") as mocked:
            llm.rank_product_ideas(IDEAS, niche="đồ bếp")
        prompt = mocked.call_args[0][0]
        self.assertIn('"Máy hút bụi mini"', prompt)
        self.assertIn("STOCK footage", prompt)
        self.assertIn("impulse-buy price point", prompt)

    def test_empty_ideas_short_circuit(self):
        self.assertEqual(llm.rank_product_ideas([]), [])
        self.assertEqual(llm.rank_product_ideas([{"reason": "no name"}]), [])


if __name__ == "__main__":
    unittest.main()
