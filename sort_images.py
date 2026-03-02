#!/usr/bin/env python3
"""
Sort captured training images into class folders for model training.

Reads capture_log.json files from training_data/ directories and copies
images into an ImageFolder structure based on the model's classification
and a confidence threshold.

Usage:
    python sort_images.py                                # Defaults: 80% confidence
    python sort_images.py --confidence 90                # Only high-confidence images
    python sort_images.py --input training_data/nighttime --output ./images
    python sort_images.py --daytime                      # Include daytime images

Output structure (ready for train_model.py):
    images/
    +-- clear/
    |   +-- 20260302_031500.jpg
    +-- cloudy/
        +-- 20260302_032000.jpg
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def sort_images(
    input_dirs: list[Path],
    output_dir: Path,
    min_confidence: float,
    dry_run: bool = False,
) -> None:
    """Sort images from capture logs into class folders.

    Args:
        input_dirs: Directories containing capture_log.json and images.
        output_dir: Output directory for sorted images.
        min_confidence: Minimum confidence threshold (0-100).
        dry_run: If True, print what would be done without copying.
    """
    total = 0
    copied = 0
    skipped_low_confidence = 0
    skipped_no_classification = 0

    for input_dir in input_dirs:
        log_path = input_dir / "capture_log.json"
        if not log_path.exists():
            print(f"Warning: No capture_log.json in {input_dir}, skipping")
            continue

        entries = json.loads(log_path.read_text())
        print(f"Processing {log_path}: {len(entries)} entries")

        for entry in entries:
            total += 1
            filename = entry.get("filename")
            class_name = entry.get("class_name")
            confidence = entry.get("confidence")

            if not filename:
                continue

            # Skip entries without classification
            if class_name is None or confidence is None:
                skipped_no_classification += 1
                continue

            # Skip low-confidence images
            if confidence < min_confidence:
                skipped_low_confidence += 1
                continue

            # Determine target folder
            class_folder = class_name.lower()
            target_dir = output_dir / class_folder
            source_path = input_dir / filename
            target_path = target_dir / filename

            if not source_path.exists():
                print(f"  Warning: {source_path} not found, skipping")
                continue

            if dry_run:
                print(f"  Would copy: {filename} -> {class_folder}/ ({confidence:.1f}%)")
            else:
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)

            copied += 1

    # Summary
    print()
    print(f"{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  Total entries:            {total}")
    print(f"  Copied:                   {copied}")
    print(f"  Skipped (low confidence): {skipped_low_confidence}")
    print(f"  Skipped (no detection):   {skipped_no_classification}")

    if not dry_run and copied > 0:
        # Show class distribution
        print()
        print("Class distribution:")
        for class_dir in sorted(output_dir.iterdir()):
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*")))
                print(f"  {class_dir.name}: {count} images")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sort captured training images into class folders",
    )
    parser.add_argument(
        "--input", type=Path, default=Path("training_data"),
        help="Input directory (default: training_data/)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("images"),
        help="Output directory for sorted images (default: images/)",
    )
    parser.add_argument(
        "--confidence", type=float, default=80.0,
        help="Minimum confidence threshold, 0-100 (default: 80)",
    )
    parser.add_argument(
        "--daytime", action="store_true",
        help="Include daytime images (default: nighttime only)",
    )
    parser.add_argument(
        "--nighttime-only", action="store_true",
        help="Only include nighttime images (default behavior)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Include both daytime and nighttime images",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without copying files",
    )
    args = parser.parse_args()

    # Determine which directories to process
    input_dirs = []
    base = args.input

    if args.all:
        # Both daytime and nighttime
        for subdir in ["nighttime", "daytime"]:
            d = base / subdir
            if d.exists():
                input_dirs.append(d)
    elif args.daytime:
        # Daytime only
        d = base / "daytime"
        if d.exists():
            input_dirs.append(d)
    else:
        # Nighttime only (default)
        d = base / "nighttime"
        if d.exists():
            input_dirs.append(d)

    # Also check if input_dir itself has a capture_log.json (flat structure)
    if not input_dirs and (base / "capture_log.json").exists():
        input_dirs.append(base)

    if not input_dirs:
        print(f"Error: No capture data found in {base}/")
        print("Expected subdirectories: nighttime/ and/or daytime/")
        sys.exit(1)

    print(f"Input directories: {[str(d) for d in input_dirs]}")
    print(f"Output directory:  {args.output}")
    print(f"Confidence threshold: {args.confidence}%")
    print()

    sort_images(
        input_dirs=input_dirs,
        output_dir=args.output,
        min_confidence=args.confidence,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
