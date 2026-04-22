#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  step3_realtime.py
#
#  Real-time Naruto hand sign detector.
#  - Resizable OpenCV window (drag corners freely)
#  - All HUD elements scale with window size
#  - "zero" pose instantly resets the sequence buffer
#
#  Keys:  Q/ESC=quit  C=clear  S=screenshot
# ─────────────────────────────────────────────────────────────
import argparse
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from utils.constants         import (COLOUR_LANDMARK, COLOUR_CONNECTION,
                                     COLOUR_HUD_BG, COLOUR_WHITE,
                                     COLOUR_YELLOW, COLOUR_GREY)
from utils.landmark_utils    import (get_hands_solution, extract_landmarks,
                                     draw_landmarks, draw_bounding_box)
from utils.sequence_detector import SequenceDetector

RESET_SIGNS  = {"zero"}   # these signs instantly clear the sequence buffer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     default="models/ensemble.pkl")
    p.add_argument("--camera",    type=int,   default=0)
    p.add_argument("--width",     type=int,   default=1280)
    p.add_argument("--height",    type=int,   default=720)
    p.add_argument("--conf",      type=float, default=0.70)
    p.add_argument("--sign_conf", type=float, default=0.60)
    return p.parse_args()


# ── Scaled drawing helpers ────────────────────────────────────
# All positions/sizes are expressed as fractions of frame dims,
# then multiplied by (w, h) so they stay correct at any window size.

def draw_hud_bg(frame, x, y, w_box, h_box, alpha=0.55):
    ov = frame.copy()
    cv2.rectangle(ov, (x, y), (x + w_box, y + h_box), COLOUR_HUD_BG, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


def put_text(frame, text, pos, scale=0.6, colour=COLOUR_WHITE,
             thickness=1, font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(frame, text, pos, font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, scale, colour,    thickness,     cv2.LINE_AA)


_pil_font_cache: dict = {}

def _get_pil_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _pil_font_cache:
        for path in [
            "C:/Windows/Fonts/seguiemj.ttf",   # Segoe UI Emoji (Windows)
            "C:/Windows/Fonts/segoeui.ttf",
        ]:
            try:
                _pil_font_cache[size] = ImageFont.truetype(path, size)
                break
            except OSError:
                pass
        else:
            _pil_font_cache[size] = ImageFont.load_default()
    return _pil_font_cache[size]


def put_text_unicode(frame, text: str, colour: tuple, font_size: int = 36,
                     cx: int | None = None, cy: int | None = None) -> None:
    """Draw Unicode/emoji text centred at (cx, cy) onto a BGR numpy frame."""
    img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _get_pil_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    h, w = frame.shape[:2]
    x = (w - tw) // 2 if cx is None else cx - tw // 2
    y = (h - th) // 2 if cy is None else cy - th // 2
    rgb = (colour[2], colour[1], colour[0])
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
    draw.text((x,     y    ), text, font=font, fill=rgb)
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_conf_bar(frame, x, y, conf, width, height):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
    bw  = int(width * conf)
    col = (0, 200, 80) if conf >= 0.75 else (0, 180, 220) if conf >= 0.50 else (0, 100, 220)
    cv2.rectangle(frame, (x, y), (x + bw, y + height), col, -1)


def draw_hold_ring(frame, cx, cy, progress, radius):
    angle = int(360 * progress)
    col   = (0, 220, 120) if progress < 1.0 else (0, 255, 200)
    cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, angle, col, 3, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (radius, radius),   0, 0, 360,   (80, 80, 80), 1, cv2.LINE_AA)


def draw_jutsu_overlay(frame, name, colour, scale):
    h, w = frame.shape[:2]
    ov   = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), colour, -1)
    cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
    bh = int(h * 0.12)
    by = (h - bh) // 2
    draw_hud_bg(frame, 0, by, w, bh, alpha=0.75)
    font_size = max(24, int(36 * scale))
    put_text_unicode(frame, name, colour, font_size=font_size, cy=by + bh // 2)


def draw_sequence_strip(frame, sequence, scale):
    h, w = frame.shape[:2]
    strip_h = int(h * 0.055)
    draw_hud_bg(frame, 0, 8, w, strip_h, alpha=0.5)
    x     = 12
    s     = max(0.38, scale * 0.55)
    for i, sign in enumerate(sequence):
        label = sign + (" -> " if i < len(sequence) - 1 else "")
        put_text(frame, label, (x, 8 + int(strip_h * 0.72)), scale=s, colour=COLOUR_YELLOW)
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, s, 1)
        x += lw + 4


def draw_zero_flash(frame):
    """Brief red tint when zero resets the sequence."""
    h, w = frame.shape[:2]
    ov   = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 180), -1)
    cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)
    put_text(frame, "SEQUENCE RESET",
             (w // 2 - 120, h // 2),
             scale=1.0, colour=(0, 80, 255), thickness=2)


# ── Main ──────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    bundle = joblib.load(args.model)
    model  = bundle["model"]
    le     = bundle["label_encoder"]
    print(f"  Classes ({len(le.classes_)}): {list(le.classes_)}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam {args.camera}")
        sys.exit(1)

    # ── Resizable window ──────────────────────────────────────
    win = "Naruto Hand Sign Detector"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)          # WINDOW_NORMAL = freely resizable
    cv2.resizeWindow(win, args.width, args.height)   # initial size

    detector     = SequenceDetector()
    fps_buf      = []
    t_prev       = time.time()
    zero_flash   = 0   # countdown frames for reset flash

    with get_hands_solution(
        static_image_mode=False,
        min_detection_confidence=args.conf,
        min_tracking_confidence=0.6,
    ) as hands:
        print("\nRunning...  Q/ESC=quit  C=clear  S=screenshot")
        print("Tip: drag window corners to resize freely.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # Scale factor relative to a 1280-wide reference
            scale = w / 1280.0

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            draw_landmarks(frame, results, COLOUR_LANDMARK, COLOUR_CONNECTION)
            draw_bounding_box(frame, results)

            # ── Predict ────────────────────────────────────
            feats      = extract_landmarks(results)
            sign_label = None
            sign_conf  = 0.0
            top_probs  = []
            is_zero    = False

            if feats is not None:
                proba     = model.predict_proba([feats])[0]
                top_idx   = np.argsort(proba)[::-1][:3]
                top_probs = [(le.classes_[i], proba[i]) for i in top_idx]
                sign_conf = top_probs[0][1]

                if sign_conf >= args.sign_conf:
                    predicted = top_probs[0][0]
                    if predicted in RESET_SIGNS:
                        is_zero = True
                        # Reset sequence immediately on confident zero detection
                        if detector.get_sequence():
                            detector.reset_sequence()
                            zero_flash = 12   # show flash for 12 frames
                    else:
                        sign_label = predicted

            detector.update(sign_label)

            # ── Zero reset flash ───────────────────────────
            if zero_flash > 0:
                draw_zero_flash(frame)
                zero_flash -= 1

            # ── Hold progress ring ─────────────────────────
            elif feats is not None and sign_label is not None:
                progress = detector.get_hold_progress()
                if progress > 0:
                    ring_r = int(30 * scale)
                    draw_hold_ring(frame, w - int(55 * scale), h - int(55 * scale),
                                   progress, ring_r)

            # ── Jutsu overlay ──────────────────────────────
            if detector.jutsu_active() and zero_flash == 0:
                name, emoji, colour = detector.current_jutsu()
                draw_jutsu_overlay(frame, f"{emoji}  {name}  {emoji}", colour, scale)

            # ── Sequence strip ─────────────────────────────
            seq = detector.get_sequence()
            if seq and zero_flash == 0:
                draw_sequence_strip(frame, seq, scale)

            # ── Side HUD ──────────────────────────────────
            hud_x  = 10
            hud_y  = int(h * 0.055) + 16
            hud_w  = int(240 * scale)
            hud_h  = int(145 * scale)
            s_main = max(0.45, scale * 0.80)
            s_sub  = max(0.30, scale * 0.45)
            s_top3 = max(0.28, scale * 0.42)

            draw_hud_bg(frame, hud_x, hud_y, hud_w, hud_h)

            if feats is not None and top_probs:
                top_name, top_conf = top_probs[0]
                if is_zero:
                    display = "-- zero --"
                    col     = COLOUR_GREY
                else:
                    display = top_name.upper()
                    col     = COLOUR_WHITE

                line1_y = hud_y + int(hud_h * 0.22)
                put_text(frame, display, (hud_x + 8, line1_y),
                         scale=s_main, colour=col, thickness=2)

                bar_y = hud_y + int(hud_h * 0.36)
                bar_h = max(5, int(8 * scale))
                draw_conf_bar(frame, hud_x + 8, bar_y, top_conf,
                              width=hud_w - 16, height=bar_h)

                pct_y = hud_y + int(hud_h * 0.52)
                put_text(frame, f"{top_conf * 100:.1f}%", (hud_x + 8, pct_y),
                         scale=s_sub, colour=COLOUR_GREY)

                for i, (nm, pb) in enumerate(top_probs[1:3], 1):
                    ty = hud_y + int(hud_h * (0.62 + i * 0.18))
                    put_text(frame, f"{nm}: {pb * 100:.0f}%",
                             (hud_x + 8, ty), scale=s_top3, colour=COLOUR_GREY)
            else:
                put_text(frame, "No hand detected",
                         (hud_x + 8, hud_y + int(hud_h * 0.35)),
                         scale=s_sub, colour=COLOUR_GREY)

            # ── FPS ────────────────────────────────────────
            t_now = time.time()
            fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now
            if len(fps_buf) > 20:
                fps_buf.pop(0)
            fps_s = max(0.3, scale * 0.5)
            put_text(frame, f"FPS {sum(fps_buf)/len(fps_buf):.0f}",
                     (w - int(90 * scale), h - int(15 * scale)),
                     scale=fps_s, colour=COLOUR_GREY)

            hint_s = max(0.28, scale * 0.40)
            put_text(frame, "[Q] Quit  [C] Clear  [S] Screenshot  |  zero = reset",
                     (10, h - int(12 * scale)), scale=hint_s, colour=COLOUR_GREY)

            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                detector.reset_sequence()
                zero_flash = 12
                print("Sequence cleared.")
            elif key == ord("s"):
                path = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(path, frame)
                print(f"Screenshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
