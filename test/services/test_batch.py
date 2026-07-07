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


class TestParseBatchItems(unittest.TestCase):
    def test_plain_line_has_empty_extras(self):
        self.assertEqual(
            batch.parse_batch_items("Nồi chiên không dầu"),
            [
                {
                    "subject": "Nồi chiên không dầu",
                    "price": "",
                    "code": "",
                    "link": "",
                }
            ],
        )

    def test_full_line_in_canonical_order(self):
        items = batch.parse_batch_items(
            "Áo thun nam | 199k | SALE50 | https://s.shopee.vn/abc"
        )
        self.assertEqual(
            items,
            [
                {
                    "subject": "Áo thun nam",
                    "price": "199k",
                    "code": "SALE50",
                    "link": "https://s.shopee.vn/abc",
                }
            ],
        )

    def test_extras_are_recognized_in_any_order(self):
        items = batch.parse_batch_items(
            "Máy hút bụi | https://ví-dụ.vn/x | 1.299.000đ"
        )
        self.assertEqual(items[0]["link"], "https://ví-dụ.vn/x")
        self.assertEqual(items[0]["price"], "1.299.000đ")
        self.assertEqual(items[0]["code"], "")

    def test_price_shapes(self):
        for price in ["199k", "1.299.000đ", "$15.99", "15.99 USD", "30%", "12"]:
            items = batch.parse_batch_items(f"Sản phẩm | {price}")
            self.assertEqual(items[0]["price"], price, price)

    def test_code_with_digits_is_not_a_price(self):
        items = batch.parse_batch_items("Sản phẩm | 50OFF")
        self.assertEqual(items[0]["code"], "50OFF")
        self.assertEqual(items[0]["price"], "")

    def test_www_link_and_first_match_wins(self):
        items = batch.parse_batch_items(
            "Sản phẩm | www.shopee.vn/a | https://shopee.vn/b | GIAM30 | MA2"
        )
        self.assertEqual(items[0]["link"], "www.shopee.vn/a")
        self.assertEqual(items[0]["code"], "GIAM30")

    def test_empty_extras_and_list_prefixes(self):
        items = batch.parse_batch_items("1. Đèn ngủ |  | SALE10 |")
        self.assertEqual(items[0]["subject"], "Đèn ngủ")
        self.assertEqual(items[0]["code"], "SALE10")

    def test_dedupes_by_subject_and_caps(self):
        text = "Son dưỡng | 99k\nSON DƯỠNG | 89k\n" + "\n".join(
            f"Sản phẩm {i}" for i in range(20)
        )
        items = batch.parse_batch_items(text)
        self.assertEqual(len(items), batch.MAX_BATCH_ITEMS)
        self.assertEqual(items[0], {
            "subject": "Son dưỡng", "price": "99k", "code": "", "link": ""
        })

    def test_has_batch_extras(self):
        self.assertFalse(
            batch.has_batch_extras(batch.parse_batch_items("Chỉ chủ đề")[0])
        )
        self.assertTrue(
            batch.has_batch_extras(batch.parse_batch_items("Sản phẩm | 99k")[0])
        )


class TestAppendIdeasToBatch(unittest.TestCase):
    def test_appends_products_to_empty_text(self):
        ideas = [{"product": "Máy hút bụi mini"}, {"product": "Son dưỡng môi"}]
        self.assertEqual(
            batch.append_ideas_to_batch("", ideas),
            "Máy hút bụi mini\nSon dưỡng môi",
        )

    def test_keeps_existing_text_verbatim_and_appends_below(self):
        existing = "1. Bình giữ nhiệt | 199k | https://s.net/a\n"
        merged = batch.append_ideas_to_batch(existing, [{"product": "Son dưỡng môi"}])
        self.assertEqual(
            merged, "1. Bình giữ nhiệt | 199k | https://s.net/a\nSon dưỡng môi"
        )

    def test_dedupes_against_existing_subjects_case_insensitively(self):
        # the existing line carries extras — dedupe must match its SUBJECT
        existing = "son dưỡng môi | 99k | CODE9"
        merged = batch.append_ideas_to_batch(
            existing, [{"product": "Son Dưỡng Môi"}, {"product": "Kẹp tóc"}]
        )
        self.assertEqual(merged, "son dưỡng môi | 99k | CODE9\nKẹp tóc")

    def test_dedupes_within_ideas_and_skips_blank_products(self):
        ideas = [
            {"product": "Kẹp tóc"},
            {"product": "kẹp tóc"},
            {"product": ""},
            {},
            None,
            {"product": "   "},
        ]
        self.assertEqual(batch.append_ideas_to_batch(None, ideas), "Kẹp tóc")

    def test_flattens_pipes_so_names_never_become_extras(self):
        merged = batch.append_ideas_to_batch("", [{"product": "Combo 2|1 khăn giấy"}])
        self.assertEqual(merged, "Combo 2 1 khăn giấy")
        self.assertEqual(
            batch.parse_batch_items(merged)[0]["subject"], "Combo 2 1 khăn giấy"
        )

    def test_nothing_new_returns_existing_unchanged_sans_trailing_newline(self):
        self.assertEqual(batch.append_ideas_to_batch("Kẹp tóc\n", [{"product": "kẹp tóc"}]), "Kẹp tóc")
        self.assertEqual(batch.append_ideas_to_batch("", []), "")

    def test_dedupes_beyond_the_batch_cap(self):
        # 12 existing lines > MAX_BATCH_ITEMS: line 12 must still block a dupe
        existing = "\n".join(f"Sản phẩm {i}" for i in range(1, 13))
        merged = batch.append_ideas_to_batch(existing, [{"product": "sản phẩm 12"}])
        self.assertEqual(merged, existing)


class TestFormatBatchCopy(unittest.TestCase):
    def test_full_bundle(self):
        block = batch.format_batch_copy(
            {
                "title": "Nồi chiên xịn",
                "caption": "Giòn rụm không dầu mỡ!",
                "hashtags": ["#noichien", "#bepnho"],
            },
            [{"comment": "Link ở đây nha 👇", "cta": "Chốt đơn ngay"}],
        )
        self.assertEqual(
            block.splitlines(),
            [
                "Title: Nồi chiên xịn",
                "Caption: Giòn rụm không dầu mỡ!",
                "Hashtags: #noichien #bepnho",
                "Pinned: Link ở đây nha 👇",
            ],
        )

    def test_missing_pieces_are_skipped(self):
        self.assertEqual(batch.format_batch_copy({}, []), "")
        self.assertEqual(batch.format_batch_copy(None, None), "")
        block = batch.format_batch_copy(
            {"caption": "Chỉ caption", "hashtags": []},
            [{"comment": "", "cta": "Mua ngay"}],
        )
        self.assertEqual(block.splitlines(), ["Caption: Chỉ caption", "Pinned: Mua ngay"])


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

    def test_cta_block_is_indented_under_its_item(self):
        results = [
            {
                "subject": "Nồi chiên",
                "videos": ["/tasks/a/final-1.mp4"],
                "error": "",
                "cta": "🛒 Nồi chiên — 199k\n👉 Link: https://s.shopee.vn/abc",
            },
            {"subject": "Son dưỡng", "videos": [], "error": "no materials"},
        ]
        summary = batch.summarize_batch_results(results)
        self.assertIn("   🛒 Nồi chiên — 199k", summary)
        self.assertIn("   👉 Link: https://s.shopee.vn/abc", summary)
        # Items without a cta key render exactly like before.
        self.assertIn("2. [FAILED] Son dưỡng (no materials)", summary)

    def test_copy_block_renders_after_cta(self):
        results = [
            {
                "subject": "Nồi chiên",
                "videos": ["/t/a.mp4"],
                "error": "",
                "cta": "🛒 Nồi chiên — 199k",
                "copy": "Caption: Giòn rụm!\nPinned: Link ở đây",
            }
        ]
        summary = batch.summarize_batch_results(results)
        self.assertIn("   Caption: Giòn rụm!", summary)
        self.assertIn("   Pinned: Link ở đây", summary)
        self.assertLess(summary.index("🛒"), summary.index("Caption:"))


if __name__ == "__main__":
    unittest.main()
