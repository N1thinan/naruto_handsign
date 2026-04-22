#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  test_model.py  –  Quick sanity check on a single image
#
#  Run:
#    python test_model.py --image path/to/hand.jpg --model models/ensemble.pkl
# ─────────────────────────────────────────────────────────────
import argparse
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.landmark_utils import get_hands_solution, extract_landmarks_from_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="models/ensemble.pkl")
    return p.parse_args()


def main():
    args   = parse_args()
    bundle = joblib.load(args.model)
    model  = bundle["model"]
    le     = bundle["label_encoder"]

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] Cannot read image: {args.image}")
        sys.exit(1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with get_hands_solution(static_image_mode=True) as hands:
        feats = extract_landmarks_from_image(rgb, hands)

    if feats is None:
        print("No hand detected in image.")
        sys.exit(0)

    proba    = model.predict_proba([feats])[0]
    top_idx  = np.argsort(proba)[::-1][:5]

    print("\n── Top-5 predictions ──────────────────────────────")
    for rank, i in enumerate(top_idx, 1):
        bar = "█" * int(proba[i] * 30)
        print(f"  {rank}. {le.classes_[i]:<10}  {proba[i]*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
