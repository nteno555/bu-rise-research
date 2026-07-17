
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
DEFAULT_SRC = "cork_oak_dataset"
DEFAULT_DST = "cork_oak_dataset_standardized"
DEFAULT_SIZE = 224
OUTPUT_EXT = ".jpg"
OUTPUT_QUALITY = 95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize bark-disease images to a uniform size (non-destructive)."
    )
    parser.add_argument(
        "--src",
        type=str,
        default=DEFAULT_SRC,
        help=f"Root input dataset directory (default: {DEFAULT_SRC})",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=DEFAULT_DST,
        help=f"Root output directory (default: {DEFAULT_DST})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"Target square size in pixels (default: {DEFAULT_SIZE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing any files.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip images that already exist in the destination.",
    )
    return parser.parse_args()


def collect_images(src_root: Path) -> list[Path]:
    images = [
        p for p in src_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    return sorted(images)


def print_dataset_summary(src_root: Path, images: list[Path]) -> None:
    print("\n" + "=" * 60)
    print(f"  Dataset summary — {src_root}")
    print("=" * 60)

    class_counts: dict[str, int] = defaultdict(int)
    size_histogram: dict[str, int] = defaultdict(int)

    for img_path in images:
        parts = img_path.relative_to(src_root).parts
        split = parts[0] if len(parts) > 1 else "root"
        cls   = parts[1] if len(parts) > 2 else "unknown"
        key   = f"{split}/{cls}"
        class_counts[key] += 1

        try:
            with Image.open(img_path) as img:
                bucket = f"{img.width}x{img.height}"
                size_histogram[bucket] += 1
        except Exception:
            size_histogram["CORRUPT"] += 1

    print("\nPer-split / per-class counts:")
    for key, count in sorted(class_counts.items()):
        print(f"  {key:<45} {count:>4} images")

    print("\nSize distribution (top 15):")
    for size, count in sorted(size_histogram.items(), key=lambda x: -x[1])[:15]:
        print(f"  {size:<20} {count:>4} images")
    print("=" * 60 + "\n")


def standardize_image(
    src_path: Path,
    dst_path: Path,
    target_size: int,
    dry_run: bool,
    no_overwrite: bool,
) -> str:

    if no_overwrite and dst_path.exists():
        return "skipped"

    if dry_run:
        return "dry_run"

    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")

            img = img.resize((target_size, target_size), Image.LANCZOS)

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_path, "JPEG", quality=OUTPUT_QUALITY, optimize=True)
            return "written"

    except (UnidentifiedImageError, OSError) as e:
        print(f"  [WARN] Could not process {src_path}: {e}")
        return "corrupt"


def main() -> None:
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    target_size: int = args.size

    if not src_root.exists():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Standardizing images")
    print(f"  Source : {src_root.resolve()}")
    print(f"  Output : {dst_root.resolve()}")
    print(f"  Size   : {target_size}×{target_size} px → {OUTPUT_EXT}")

    images = collect_images(src_root)
    if not images:
        print("[WARN] No images found in source directory.")
        return

    print_dataset_summary(src_root, images)

    dst_paths = [
        dst_root / img.relative_to(src_root).with_suffix(OUTPUT_EXT)
        for img in images
    ]

    stats: dict[str, int] = defaultdict(int)

    for src_path, dst_path in tqdm(
        zip(images, dst_paths), total=len(images), desc="Processing"
    ):
        result = standardize_image(
            src_path, dst_path, target_size, args.dry_run, args.no_overwrite
        )
        stats[result] += 1

    print("\n" + "=" * 60)
    print("  Standardization complete")
    print("=" * 60)
    if args.dry_run:
        print(f"  Would process : {stats['dry_run']} images (dry run — nothing written)")
    else:
        print(f"  Written       : {stats['written']}")
        print(f"  Skipped       : {stats['skipped']} (already existed)")
        print(f"  Corrupt/Error : {stats['corrupt']}")
        print(f"\n  Output saved to: {dst_root.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
