import os
import sys
import unittest

root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.services.product_media import weave_product_items


class TestWeaveProductItems(unittest.TestCase):
    def test_both_empty(self):
        self.assertEqual(weave_product_items([], []), [])

    def test_no_product_returns_stock_unchanged(self):
        self.assertEqual(weave_product_items([], ["s1", "s2"]), ["s1", "s2"])

    def test_no_stock_returns_products_only(self):
        self.assertEqual(weave_product_items(["p1", "p2"], []), ["p1", "p2"])

    def test_single_product_opens_the_video(self):
        self.assertEqual(
            weave_product_items(["p1"], ["s1", "s2", "s3"]),
            ["p1", "s1", "s2", "s3"],
        )

    def test_multiple_products_spread_evenly(self):
        self.assertEqual(
            weave_product_items(
                ["p1", "p2", "p3"], ["s1", "s2", "s3", "s4", "s5", "s6"]
            ),
            ["p1", "s1", "s2", "p2", "s3", "s4", "p3", "s5", "s6"],
        )

    def test_first_item_is_always_a_product(self):
        woven = weave_product_items(["p1", "p2"], ["s1", "s2", "s3", "s4", "s5"])
        self.assertEqual(woven[0], "p1")
        self.assertEqual(sorted(woven), sorted(["p1", "p2", "s1", "s2", "s3", "s4", "s5"]))

    def test_more_products_than_stock_keeps_everything(self):
        woven = weave_product_items(["p1", "p2", "p3", "p4"], ["s1"])
        self.assertEqual(woven[0], "p1")
        self.assertEqual(
            sorted(woven), sorted(["p1", "p2", "p3", "p4", "s1"])
        )

    def test_falsy_entries_are_dropped(self):
        self.assertEqual(
            weave_product_items(["", "p1", None], ["s1", "", None]),
            ["p1", "s1"],
        )

    def test_inputs_are_not_mutated(self):
        products = ["p1", "p2"]
        stock = ["s1", "s2"]
        weave_product_items(products, stock)
        self.assertEqual(products, ["p1", "p2"])
        self.assertEqual(stock, ["s1", "s2"])

    def test_works_on_arbitrary_objects(self):
        p = [{"id": "p1"}]
        s = [{"id": "s1"}]
        self.assertEqual(weave_product_items(p, s), [{"id": "p1"}, {"id": "s1"}])


if __name__ == "__main__":
    unittest.main()
