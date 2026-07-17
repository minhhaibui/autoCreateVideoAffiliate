import json
import os
import sys
import tempfile
import unittest
import uuid

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.utils import utils


class TestResolveTaskFolder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(os.rmdir, self.tmp)

    def test_valid_uuid_resolves_inside_root(self):
        task_id = str(uuid.uuid4())
        path = utils.resolve_task_folder(task_id, tasks_root=self.tmp)
        self.assertEqual(path, os.path.join(os.path.abspath(self.tmp), task_id))

    def test_hyphenless_uuid_is_normalized_to_hyphenated(self):
        task_id = uuid.uuid4()
        bare = task_id.hex
        path = utils.resolve_task_folder(bare, tasks_root=self.tmp)
        self.assertEqual(os.path.basename(path), str(task_id))

    def test_non_uuid_returns_none(self):
        self.assertIsNone(utils.resolve_task_folder("not-a-uuid", tasks_root=self.tmp))

    def test_path_traversal_attempt_returns_none(self):
        self.assertIsNone(
            utils.resolve_task_folder("../../etc/passwd", tasks_root=self.tmp)
        )

    def test_none_and_empty_return_none(self):
        self.assertIsNone(utils.resolve_task_folder(None, tasks_root=self.tmp))
        self.assertIsNone(utils.resolve_task_folder("", tasks_root=self.tmp))

    def test_default_root_is_storage_tasks(self):
        task_id = str(uuid.uuid4())
        path = utils.resolve_task_folder(task_id)
        expected_root = os.path.abspath(os.path.join(utils.storage_dir(), "tasks"))
        self.assertEqual(path, os.path.join(expected_root, task_id))

    def test_never_creates_the_folder(self):
        task_id = str(uuid.uuid4())
        path = utils.resolve_task_folder(task_id, tasks_root=self.tmp)
        self.assertFalse(os.path.exists(path))


class TestGetResponse(unittest.TestCase):
    def test_status_only(self):
        self.assertEqual(utils.get_response(200), {"status": 200})

    def test_data_and_message_included_when_truthy(self):
        self.assertEqual(
            utils.get_response(400, {"k": 1}, "bad"),
            {"status": 400, "data": {"k": 1}, "message": "bad"},
        )


class TestToJson(unittest.TestCase):
    def test_vietnamese_text_is_not_ascii_escaped(self):
        out = utils.to_json({"msg": "nồi chiên không dầu"})
        self.assertIn("nồi chiên không dầu", out)

    def test_bytes_are_masked(self):
        out = utils.to_json({"blob": b"\x00\x01"})
        self.assertIn("*** binary data ***", out)

    def test_object_with_dict_is_serialized(self):
        class Thing:
            def __init__(self):
                self.name = "x"

        self.assertEqual(json.loads(utils.to_json(Thing())), {"name": "x"})

    def test_unserializable_value_becomes_null(self):
        out = utils.to_json({"s": {1, 2}})
        self.assertEqual(json.loads(out), {"s": None})


class TestGetUuid(unittest.TestCase):
    def test_default_is_valid_hyphenated_uuid(self):
        value = utils.get_uuid()
        self.assertEqual(str(uuid.UUID(value)), value)

    def test_remove_hyphen(self):
        self.assertNotIn("-", utils.get_uuid(remove_hyphen=True))


class TestTimeConvert(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(utils.time_convert_seconds_to_hmsm(0), "00:00:00,000")

    def test_hours_minutes_seconds_millis(self):
        self.assertEqual(utils.time_convert_seconds_to_hmsm(3661.5), "01:01:01,500")

    def test_just_under_a_minute(self):
        self.assertEqual(utils.time_convert_seconds_to_hmsm(59.999), "00:00:59,999")


class TestTextToSrt(unittest.TestCase):
    def test_block_structure(self):
        srt = utils.text_to_srt(3, "Xin chào", 1.5, 2.25)
        lines = srt.splitlines()
        self.assertEqual(lines[0], "3")
        self.assertEqual(lines[1], "00:00:01,500 --> 00:00:02,250")
        self.assertEqual(lines[2], "Xin chào")


class TestPunctuationHelpers(unittest.TestCase):
    def test_str_contains_punctuation(self):
        self.assertTrue(utils.str_contains_punctuation("xin chào!"))
        self.assertFalse(utils.str_contains_punctuation("xin chào"))

    def test_split_basic_sentences(self):
        self.assertEqual(
            utils.split_string_by_punctuations("Câu một. Câu hai!"),
            ["Câu một", "Câu hai"],
        )

    def test_decimal_dot_is_not_a_break(self):
        self.assertEqual(
            utils.split_string_by_punctuations("phí 2.5% thôi"), ["phí 2.5% thôi"]
        )

    def test_thousands_comma_is_not_a_break(self):
        self.assertEqual(
            utils.split_string_by_punctuations("giá 1,000 đồng"), ["giá 1,000 đồng"]
        )

    def test_newline_splits_and_empties_are_dropped(self):
        self.assertEqual(
            utils.split_string_by_punctuations("dòng một\n\ndòng hai"),
            ["dòng một", "dòng hai"],
        )


class TestNormalizeScriptForSubtitleMatching(unittest.TestCase):
    def test_underscores_removed(self):
        self.assertEqual(
            utils.normalize_script_for_subtitle_matching("giảm _sốc_ hôm nay"),
            "giảm sốc hôm nay",
        )

    def test_markdown_separator_lines_removed(self):
        self.assertEqual(
            utils.normalize_script_for_subtitle_matching("mở đầu\n---\nkết thúc"),
            "mở đầu\nkết thúc",
        )

    def test_none_becomes_empty_string(self):
        self.assertEqual(utils.normalize_script_for_subtitle_matching(None), "")


class TestSmallHelpers(unittest.TestCase):
    def test_md5_known_value(self):
        self.assertEqual(utils.md5("abc"), "900150983cd24fb0d6963f7d28e17f72")

    def test_parse_extension_lowercases(self):
        self.assertEqual(utils.parse_extension("Video.MP4"), "mp4")

    def test_parse_extension_takes_last_suffix(self):
        self.assertEqual(utils.parse_extension("a.tar.gz"), "gz")

    def test_parse_extension_no_suffix(self):
        self.assertEqual(utils.parse_extension("noext"), "")

    def test_storage_dir_points_into_repo_without_creating(self):
        d = utils.storage_dir("definitely-missing-subdir-xyz")
        self.assertFalse(os.path.exists(d))
        self.assertTrue(d.endswith(os.path.join("storage", "definitely-missing-subdir-xyz")))

    def test_resource_dir_points_into_repo(self):
        self.assertTrue(utils.resource_dir("fonts").endswith(os.path.join("resource", "fonts")))


if __name__ == "__main__":
    unittest.main()
