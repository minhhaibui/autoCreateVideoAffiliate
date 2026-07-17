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

from app.services import product_library


class TestTokens(unittest.TestCase):
    def test_diacritics_and_case_are_stripped(self):
        self.assertEqual(
            product_library._tokens("Nồi CHIÊN không dầu"),
            {"noi", "chien", "khong", "dau"},
        )

    def test_d_bar_maps_to_d(self):
        self.assertEqual(product_library._tokens("đồ gia dụng"), {"do", "gia", "dung"})

    def test_punctuation_splits(self):
        self.assertEqual(
            product_library._tokens("máy-xay(mini)"), {"may", "xay", "mini"}
        )


class TestMatchFolder(unittest.TestCase):
    FOLDERS = ["noi chien khong dau", "may xay sinh to", "binh giu nhiet"]

    def test_full_subject_matches_its_folder(self):
        self.assertEqual(
            product_library.match_folder(
                "Nồi chiên không dầu 5L chính hãng", self.FOLDERS
            ),
            "noi chien khong dau",
        )

    def test_unrelated_subject_matches_nothing(self):
        self.assertEqual(
            product_library.match_folder("Áo thun nam cotton", self.FOLDERS), ""
        )

    def test_partial_overlap_below_threshold_is_rejected(self):
        # only 1 of 4 folder tokens present -> ratio 0.25 < 0.6
        self.assertEqual(
            product_library.match_folder("nồi cơm điện", self.FOLDERS), ""
        )

    def test_most_specific_folder_wins(self):
        folders = ["may xay", "may xay sinh to mini"]
        self.assertEqual(
            product_library.match_folder(
                "Máy xay sinh tố mini cầm tay", folders
            ),
            "may xay sinh to mini",
        )

    def test_empty_product_name_matches_nothing(self):
        self.assertEqual(product_library.match_folder("", self.FOLDERS), "")


class TestStageProductMedia(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.library = os.path.join(self.tmp.name, "product_library")
        self.staging = os.path.join(self.tmp.name, "product_media")

        def fake_storage_dir(sub_dir="", create=False):
            d = os.path.join(self.tmp.name, sub_dir) if sub_dir else self.tmp.name
            if create and not os.path.exists(d):
                os.makedirs(d)
            return d

        patcher = patch.object(
            product_library.utils, "storage_dir", side_effect=fake_storage_dir
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def seed(self, folder, files):
        d = os.path.join(self.library, folder)
        os.makedirs(d, exist_ok=True)
        for name in files:
            with open(os.path.join(d, name), "wb") as f:
                f.write(b"fake")

    def test_no_library_folder_returns_empty_and_scaffolds(self):
        materials = product_library.stage_product_media("Nồi chiên không dầu")
        self.assertEqual(materials, [])
        self.assertTrue(
            os.path.isfile(
                os.path.join(self.library, product_library.GUIDE_FILENAME)
            )
        )

    def test_matched_folder_stages_media_into_product_media(self):
        self.seed("noi chien khong dau", ["a.jpg", "b.png"])
        materials = product_library.stage_product_media(
            "Nồi chiên không dầu 5L"
        )
        self.assertEqual(len(materials), 2)
        for m in materials:
            self.assertTrue(m.url.startswith(self.staging))
            self.assertTrue(os.path.isfile(m.url))
            self.assertEqual(m.provider, "local")

    def test_non_media_files_are_skipped(self):
        self.seed("noi chien khong dau", ["a.jpg", "notes.txt", "link.url"])
        materials = product_library.stage_product_media("nồi chiên không dầu")
        self.assertEqual(len(materials), 1)
        self.assertTrue(materials[0].url.endswith("a.jpg"))

    def test_unmatched_subject_stages_nothing(self):
        self.seed("noi chien khong dau", ["a.jpg"])
        self.assertEqual(
            product_library.stage_product_media("Áo thun nam"), []
        )

    def test_staged_names_are_prefixed_per_folder(self):
        self.seed("noi chien khong dau", ["a.jpg"])
        materials = product_library.stage_product_media("nồi chiên không dầu")
        self.assertIn("lib-noi chien khong dau-a.jpg", materials[0].url)


class TestListFolderMedia(unittest.TestCase):
    def test_media_sorted_and_non_media_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ["b.png", "a.jpg", "link.txt", "HUONG-DAN.txt", "c.mov"]:
                with open(os.path.join(tmp, name), "wb") as f:
                    f.write(b"x")
            self.assertEqual(
                product_library.list_folder_media(tmp), ["a.jpg", "b.png", "c.mov"]
            )

    def test_missing_or_empty_folder(self):
        self.assertEqual(product_library.list_folder_media(""), [])
        self.assertEqual(product_library.list_folder_media("/no/such/dir"), [])


class TestReadProductLink(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.library = os.path.join(self.tmp.name, "product_library")

        def fake_storage_dir(sub_dir="", create=False):
            d = os.path.join(self.tmp.name, sub_dir) if sub_dir else self.tmp.name
            if create and not os.path.exists(d):
                os.makedirs(d)
            return d

        patcher = patch.object(
            product_library.utils, "storage_dir", side_effect=fake_storage_dir
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def seed_link(self, folder, content):
        d = os.path.join(self.library, folder)
        os.makedirs(d, exist_ok=True)
        with open(
            os.path.join(d, product_library.LINK_FILENAME), "w", encoding="utf-8"
        ) as f:
            f.write(content)

    def test_first_nonempty_line_is_returned_stripped(self):
        self.seed_link(
            "noi chien khong dau", "\n  https://vt.tiktok.com/ZS123abc/  \nghi chú\n"
        )
        self.assertEqual(
            product_library.read_product_link("Nồi chiên không dầu 5L"),
            "https://vt.tiktok.com/ZS123abc/",
        )

    def test_missing_link_file_returns_empty(self):
        os.makedirs(os.path.join(self.library, "noi chien khong dau"))
        self.assertEqual(
            product_library.read_product_link("nồi chiên không dầu"), ""
        )

    def test_no_matching_folder_returns_empty(self):
        self.assertEqual(product_library.read_product_link("Áo thun"), "")

    def test_empty_file_returns_empty(self):
        self.seed_link("noi chien khong dau", "\n\n")
        self.assertEqual(
            product_library.read_product_link("nồi chiên không dầu"), ""
        )


class TestReportAffiliateLink(unittest.TestCase):
    def test_report_carries_library_link_and_drops_manual_note(self):
        from app.services import autopilot

        report = autopilot.format_report(
            {"product": "Nồi chiên"},
            ["/t/final-1.mp4"],
            {},
            [{"comment": "Link ở đây nha 👆"}],
            affiliate_link="https://vt.tiktok.com/ZS123abc/",
        )
        self.assertIn(
            "Affiliate link (from your product library): https://vt.tiktok.com/ZS123abc/",
            report,
        )
        self.assertNotIn("never invents links", report)

    def test_report_without_link_keeps_manual_note(self):
        from app.services import autopilot

        report = autopilot.format_report(
            {"product": "Nồi chiên"}, ["/t/final-1.mp4"], {}, []
        )
        self.assertIn("never invents links", report)
        self.assertNotIn("Affiliate link", report)


class TestTaskLibraryFallback(unittest.TestCase):
    def test_task_uses_library_when_no_manual_product_media(self):
        from app.models.schema import MaterialInfo, VideoParams
        from app.services import task as tm

        params = VideoParams(video_subject="Nồi chiên không dầu")
        staged = [MaterialInfo(provider="local", url="lib-p.jpg")]
        processed = [MaterialInfo(provider="local", url="lib-p.mp4")]
        with patch.object(
            tm.product_library, "stage_product_media", return_value=staged
        ) as mock_stage, patch.object(
            tm.video, "preprocess_video", return_value=processed
        ):
            result = tm.preprocess_product_materials(params)
        mock_stage.assert_called_once_with("Nồi chiên không dầu")
        self.assertEqual(result, ["lib-p.mp4"])

    def test_manual_upload_takes_priority_over_library(self):
        from app.models.schema import MaterialInfo, VideoParams
        from app.services import task as tm

        params = VideoParams(
            video_subject="x",
            product_materials=[MaterialInfo(provider="local", url="manual.jpg")],
        )
        with patch.object(
            tm.product_library, "stage_product_media"
        ) as mock_stage, patch.object(
            tm.video, "preprocess_video", return_value=[]
        ):
            tm.preprocess_product_materials(params)
        mock_stage.assert_not_called()

    def test_library_failure_never_fails_the_render(self):
        from app.models.schema import VideoParams
        from app.services import task as tm

        params = VideoParams(video_subject="x")
        with patch.object(
            tm.product_library,
            "stage_product_media",
            side_effect=RuntimeError("boom"),
        ):
            self.assertEqual(tm.preprocess_product_materials(params), [])


if __name__ == "__main__":
    unittest.main()
