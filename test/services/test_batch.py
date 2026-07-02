import unittest

from app.services import batch


class TestParseBatchSubjects(unittest.TestCase):
    def test_splits_strips_and_drops_empty_lines(self):
        text = "  Máy hút bụi mini \n\n   \nSon dưỡng môi\n"
        self.assertEqual(
            batch.parse_batch_subjects(text),
            ["Máy hút bụi mini", "Son dưỡng môi"],
        )

    def test_strips_common_list_prefixes(self):
        text = "1. Nồi chiên không dầu\n2) Đèn ngủ cảm ứng\n- Bình giữ nhiệt\n* Kem chống nắng\n• Tai nghe bluetooth"
        self.assertEqual(
            batch.parse_batch_subjects(text),
            [
                "Nồi chiên không dầu",
                "Đèn ngủ cảm ứng",
                "Bình giữ nhiệt",
                "Kem chống nắng",
                "Tai nghe bluetooth",
            ],
        )

    def test_keeps_numbers_that_are_part_of_the_subject(self):
        # A bare number with no separator is product text, not a list prefix.
        self.assertEqual(
            batch.parse_batch_subjects("3 món đồ bếp thông minh"),
            ["3 món đồ bếp thông minh"],
        )

    def test_dedupes_case_insensitively_preserving_order(self):
        text = "Son dưỡng môi\nSON DƯỠNG MÔI\nMáy hút bụi\nson dưỡng môi"
        self.assertEqual(
            batch.parse_batch_subjects(text),
            ["Son dưỡng môi", "Máy hút bụi"],
        )

    def test_caps_at_max_items(self):
        text = "\n".join(f"Sản phẩm {i}" for i in range(20))
        result = batch.parse_batch_subjects(text)
        self.assertEqual(len(result), batch.MAX_BATCH_ITEMS)
        self.assertEqual(result[0], "Sản phẩm 0")
        result_small = batch.parse_batch_subjects(text, max_items=3)
        self.assertEqual(len(result_small), 3)

    def test_empty_and_none_input(self):
        self.assertEqual(batch.parse_batch_subjects(""), [])
        self.assertEqual(batch.parse_batch_subjects(None), [])
        self.assertEqual(batch.parse_batch_subjects("   \n \n"), [])


class TestSummarizeBatchResults(unittest.TestCase):
    def test_mixed_results(self):
        results = [
            {"subject": "Nồi chiên", "videos": ["/tasks/a/final-1.mp4"], "error": ""},
            {"subject": "Son dưỡng", "videos": [], "error": "no materials"},
        ]
        summary = batch.summarize_batch_results(results)
        self.assertIn("1/2 OK", summary)
        self.assertIn("1. [OK] Nồi chiên", summary)
        self.assertIn("   - /tasks/a/final-1.mp4", summary)
        self.assertIn("2. [FAILED] Son dưỡng (no materials)", summary)

    def test_failure_without_error_message_gets_default_reason(self):
        summary = batch.summarize_batch_results(
            [{"subject": "Đèn ngủ", "videos": [], "error": ""}]
        )
        self.assertIn("(failed)", summary)

    def test_empty_results(self):
        self.assertEqual(batch.summarize_batch_results([]), "0/0 OK")


if __name__ == "__main__":
    unittest.main()
