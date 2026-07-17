"""Product media library: drop listing photos in a folder ONCE, every future
render of that product picks them up automatically.

Layout: storage/product_library/<folder-per-product>/*.jpg|png|mp4|mov
The folder name is matched against the (LLM-generated or user-typed) product
name with diacritic-insensitive token overlap, so the folder "noi chien khong
dau" matches the subject "Nồi chiên không dầu 5L chính hãng". Matched files
are STAGED (copied) into storage/product_media/ — the directory the render
pipeline's path-security whitelist already trusts — and returned as
MaterialInfo, ready for VideoParams.product_materials.

This is the autopilot's bridge to the real-product-media feature: the human
uploads photos once per product; every automated render after that shows the
actual product with zero manual steps.
"""

import os
import shutil
import unicodedata

from loguru import logger

from app.models.schema import MaterialInfo
from app.utils import utils

MEDIA_EXTENSIONS = ("jpg", "jpeg", "png", "mp4", "mov")
# A folder counts as a match when at least this share of ITS OWN tokens
# appear in the product name (folder names are the shorter, curated side).
MIN_MATCH_RATIO = 0.6
GUIDE_FILENAME = "HUONG-DAN.txt"
GUIDE_TEXT = """Thư mục THƯ VIỆN SẢN PHẨM (tự động dùng cho video)

Cách dùng:
1. Tạo một thư mục con cho mỗi sản phẩm, tên KHÔNG cần dấu, ví dụ:
     noi chien khong dau/
     may xay sinh to/
2. Thả ảnh/video thật của sản phẩm vào thư mục đó (jpg, png, mp4, mov).
   Nên dùng 2-4 ảnh đẹp nhất từ trang shop (ảnh có người dùng càng tốt).
3. Xong! Khi Autopilot (hoặc bạn) làm video cho sản phẩm có tên khớp,
   ảnh sẽ tự xuất hiện: mở đầu video + rải đều giữa các cảnh quay.

Lưu ý: ảnh phải rộng/cao tối thiểu 480px; tên thư mục khớp theo từ
(không phân biệt hoa thường, không cần dấu tiếng Việt).
"""


def library_dir(create: bool = False) -> str:
    return utils.storage_dir("product_library", create=create)


def ensure_library_scaffold() -> str:
    """Create the library folder + a Vietnamese how-to file so the user can
    discover the drop-point without reading docs. Idempotent."""
    root = library_dir(create=True)
    guide_path = os.path.join(root, GUIDE_FILENAME)
    if not os.path.isfile(guide_path):
        try:
            with open(guide_path, "w", encoding="utf-8") as f:
                f.write(GUIDE_TEXT)
        except OSError as e:
            logger.warning(f"could not write product library guide: {e}")
    return root


def _tokens(text: str) -> set:
    """Diacritic-insensitive, case-insensitive word set: 'Nồi chiên KHÔNG dầu'
    -> {'noi', 'chien', 'khong', 'dau'}. đ/Đ have no combining mark, map manually."""
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return {t for t in "".join(
        c if c.isalnum() else " " for c in stripped.casefold()
    ).split() if t}


def match_folder(product_name: str, folder_names) -> str:
    """Best library folder for this product name, or "".

    Score = share of the folder's tokens found in the product name. Requires
    MIN_MATCH_RATIO so 'may xay sinh to' never matches 'Nồi chiên không dầu';
    ties break toward the folder with more matching tokens (most specific).
    """
    product_tokens = _tokens(product_name)
    if not product_tokens:
        return ""
    best_name, best_key = "", (0.0, 0)
    for name in folder_names:
        folder_tokens = _tokens(name)
        if not folder_tokens:
            continue
        overlap = len(folder_tokens & product_tokens)
        ratio = overlap / len(folder_tokens)
        if ratio < MIN_MATCH_RATIO:
            continue
        key = (ratio, overlap)
        if key > best_key:
            best_name, best_key = name, key
    return best_name


def find_product_folder(product_name: str) -> str:
    """Absolute path of the matching library folder, or ""."""
    root = library_dir()
    if not os.path.isdir(root):
        return ""
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    matched = match_folder(product_name, folders)
    return os.path.join(root, matched) if matched else ""


def stage_product_media(product_name: str) -> list:
    """Copy the matched folder's media into storage/product_media (the
    render pipeline's trusted directory) and return MaterialInfo entries.
    Returns [] when there is no matching folder or no usable files —
    callers treat that as 'render without product media', never an error.
    """
    # Scaffold on every lookup (idempotent): the folder + vi guide appearing
    # in storage/ is how the user discovers the drop-point.
    ensure_library_scaffold()
    folder = find_product_folder(product_name)
    if not folder:
        return []

    staging_dir = utils.storage_dir("product_media", create=True)
    prefix = f"lib-{os.path.basename(folder)}"
    materials = []
    for name in sorted(os.listdir(folder)):
        ext = utils.parse_extension(name)
        if ext not in MEDIA_EXTENSIONS:
            continue
        src = os.path.join(folder, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(staging_dir, f"{prefix}-{name}")
        try:
            shutil.copyfile(src, dst)
        except OSError as e:
            logger.warning(f"product library: could not stage {src}: {e}")
            continue
        materials.append(MaterialInfo(provider="local", url=dst))
    if materials:
        logger.info(
            f"product library: matched folder '{os.path.basename(folder)}' for "
            f"{product_name!r} — staged {len(materials)} file(s)"
        )
    return materials
