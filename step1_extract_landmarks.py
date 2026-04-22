#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  step1_extract_landmarks.py
#
#  Reads train/ and test/ splits, extracts 42 normalised
#  landmark features with MediaPipe, saves two CSV files.
#
#  Expected dataset layout:
#    data/
#      train/
#        bird/    img001.jpg ...
#        boar/    ...
#        ...
#        zero/    ...
#      test/
#        bird/    ...
#        ...
#
#  Run:
#    python step1_extract_landmarks.py
#
#  Outputs:
#    data/landmarks_train.csv
#    data/landmarks_test.csv
# ─────────────────────────────────────────────────────────────
import argparse
import csv
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.constants      import HAND_SIGNS, NUM_FEATURES, TRAIN_DIR, TEST_DIR
from utils.landmark_utils import get_hands_solution, extract_landmarks_from_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HEADER   = [f"f{i}" for i in range(NUM_FEATURES)] + ["label"]


def process_split(split_dir: Path, out_csv: Path, min_conf: float) -> None:
    split_dir = Path(split_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    processed = skipped = missing = 0

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        with get_hands_solution(static_image_mode=True,
                                min_detection_confidence=min_conf) as hands:

            for sign_name in HAND_SIGNS:
                sign_dir = split_dir / sign_name
                if not sign_dir.exists():
                    print(f"  [WARNING] Folder not found: {sign_dir}")
                    missing += 1
                    continue

                images = [p for p in sorted(sign_dir.glob("*"))
                          if p.suffix.lower() in IMG_EXTS]

                for img_path in tqdm(images, desc=f"  {sign_name:<8}", leave=False):
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        skipped += 1
                        continue

                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    feats   = extract_landmarks_from_image(img_rgb, hands)

                    if feats is None:
                        skipped += 1
                        continue

                    writer.writerow(list(feats) + [sign_name])
                    processed += 1
                
                print(feats)
    
    print(f"  ✓ {processed} samples saved  |  {skipped} skipped (no hand detected)  |  {missing} missing folders")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir",    default=TRAIN_DIR)
    p.add_argument("--test_dir",     default=TEST_DIR)
    p.add_argument("--train_out",    default="data/landmarks_train.csv")
    p.add_argument("--test_out",     default="data/landmarks_test.csv")
    p.add_argument("--min_conf",     type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n── Extracting TRAIN split from '{args.train_dir}' ──")
    process_split(args.train_dir, Path(args.train_out), args.min_conf)
    print(f"   → {args.train_out}\n")

    print(f"── Extracting TEST split from '{args.test_dir}' ──")
    process_split(args.test_dir, Path(args.test_out), args.min_conf)
    print(f"   → {args.test_out}\n")

    print("✓ Done.")


if __name__ == "__main__":
    main()
