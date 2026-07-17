"""Weaving real product media into the stock-footage timeline.

An affiliate video that never shows the actual product converts poorly. The
user uploads real photos/clips of the product (e.g. the shop listing's photos
of a model wearing the outfit); this module decides WHERE those clips go:
the first product clip opens the video (the 2-second hook must show the real
product) and the rest are spread evenly between the stock clips so the
product keeps reappearing instead of clustering.

Pure list reordering — no I/O — so both the path-level merge (task service)
and the subclip-level pinning (video service) can share and unit-test it.
"""

import math
from typing import List, Sequence


def weave_product_items(product_items: Sequence, stock_items: Sequence) -> List:
    """Return one list: product_items[0] first, then stock_items with the
    remaining product items inserted at even intervals.

    Works on any item type (file paths, subclip objects). Falsy entries are
    dropped. With no product items the stock list is returned unchanged (and
    vice versa), so callers can apply it unconditionally.
    """
    products = [item for item in product_items if item]
    stock = [item for item in stock_items if item]
    if not products:
        return stock
    if not stock:
        return products

    result = [products[0]]
    rest = products[1:]
    if not rest:
        return result + stock

    # Split the stock into len(rest)+1 chunks and put one product clip after
    # each chunk boundary: P0 S..S P1 S..S P2 S..S — evenly spread, product
    # never opens back-to-back with itself.
    chunk = math.ceil(len(stock) / (len(rest) + 1))
    for i in range(0, len(stock), chunk):
        result.extend(stock[i : i + chunk])
        if rest:
            result.append(rest.pop(0))
    # More product clips than chunk boundaries — append what's left.
    result.extend(rest)
    return result
