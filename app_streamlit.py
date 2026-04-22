#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  app_streamlit.py  –  Browser UI
#
#  Run:
#    streamlit run app_streamlit.py -- --model models/ensemble.pkl
# ─────────────────────────────────────────────────────────────
import argparse
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from utils.constants         import HAND_SIGNS, JUTSU_DICT, COLOUR_LANDMARK, COLOUR_CONNECTION
from utils.landmark_utils    import get_hands_solution, extract_landmarks, draw_landmarks, draw_bounding_box
from utils.sequence_detector import SequenceDetector

IGNORED_SIGNS = {"zero"}

st.set_page_config(page_title="Naruto Hand Sign Detector", page_icon="🍃", layout="wide")

st.markdown("""
<style>
  .jutsu-banner {
    background: linear-gradient(135deg,#1a0a00,#3d1a00);
    border: 2px solid #ff6600; border-radius:12px;
    padding:24px; text-align:center; margin:12px 0;
  }
  .jutsu-name { font-size:2.2rem; font-weight:800; color:#ffaa00;
                text-shadow:0 0 20px #ff6600; letter-spacing:2px; }
  .sign-chip  { display:inline-block; background:#1e3a2e;
                border:1px solid #00cc66; border-radius:20px;
                padding:4px 14px; margin:3px; font-size:.9rem; color:#00ff88; }
</style>""", unsafe_allow_html=True)


def get_model_path():
    try:
        idx = sys.argv.index("--model")
        return sys.argv[idx + 1]
    except (ValueError, IndexError):
        return "models/ensemble.pkl"


@st.cache_resource
def load_bundle(path):
    b = joblib.load(path)
    return b["model"], b["label_encoder"]


if "detector" not in st.session_state:
    st.session_state.detector  = SequenceDetector()
if "hands" not in st.session_state:
    st.session_state.hands     = get_hands_solution(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ).__enter__()
if "running"    not in st.session_state: st.session_state.running    = False
if "jutsu_log"  not in st.session_state: st.session_state.jutsu_log  = []

st.title("🍃 Naruto Hand Sign Detector")

model_path = get_model_path()
try:
    model, le = load_bundle(model_path)
except Exception as e:
    st.error(f"Could not load model '{model_path}': {e}")
    st.info("Run `python step2_train_model.py` first.")
    st.stop()

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📷 Live Camera Feed")
    conf_thresh = st.slider("Min prediction confidence", 0.40, 0.95, 0.65, 0.05)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶ Start" if not st.session_state.running else "⏹ Stop"):
            st.session_state.running = not st.session_state.running
    with c2:
        if st.button("🗑 Clear sequence"): st.session_state.detector.reset_sequence()
    with c3:
        if st.button("🔃 Reset log"): st.session_state.jutsu_log.clear()
    frame_ph  = st.empty()
    status_ph = st.empty()

with col_right:
    st.subheader("🔮 Jutsu Output")
    jutsu_ph = st.empty()
    st.subheader("📋 Current Sequence")
    seq_ph   = st.empty()
    st.subheader("📜 Cast History")
    log_ph   = st.empty()
    st.subheader("📖 Known Jutsu")
    with st.expander("Show all combos"):
        for signs, (name, emoji, _) in JUTSU_DICT.items():
            st.markdown(f"`{' → '.join(signs)}` → **{emoji} {name}**")

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        st.error("Cannot open webcam.")
        st.session_state.running = False
    else:
        detector = st.session_state.detector
        hands    = st.session_state.hands
        try:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res   = hands.process(rgb)

                draw_landmarks(frame, res, COLOUR_LANDMARK, COLOUR_CONNECTION)
                draw_bounding_box(frame, res)

                feats      = extract_landmarks(res)
                sign_label = None
                sign_conf  = 0.0
                top_probs  = []

                if feats is not None:
                    proba     = model.predict_proba([feats])[0]
                    top_idx   = np.argsort(proba)[::-1][:3]
                    top_probs = [(le.classes_[i], proba[i]) for i in top_idx]
                    sign_conf = top_probs[0][1]
                    if sign_conf >= conf_thresh:
                        pred = top_probs[0][0]
                        sign_label = None if pred in IGNORED_SIGNS else pred

                was_active = detector.jutsu_active()
                detector.update(sign_label)

                if detector.jutsu_active() and not was_active:
                    name, emoji, _ = detector.current_jutsu()
                    st.session_state.jutsu_log.append(f"{emoji} {name}")

                frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                if feats is not None and top_probs:
                    top_name = top_probs[0][0]
                    if top_name in IGNORED_SIGNS:
                        status_ph.info(f"🤚 Neutral pose (zero)  |  {sign_conf*100:.1f}%")
                    elif sign_label:
                        status_ph.success(
                            f"✋ **{sign_label}** — {sign_conf*100:.1f}%  |  "
                            f"Hold: {detector.get_hold_progress()*100:.0f}%")
                    else:
                        status_ph.warning("Confidence too low")
                else:
                    status_ph.info("No hand detected")

                if detector.jutsu_active():
                    name, emoji, _ = detector.current_jutsu()
                    jutsu_ph.markdown(
                        f'<div class="jutsu-banner"><div class="jutsu-name">{emoji} {name} {emoji}</div></div>',
                        unsafe_allow_html=True)
                else:
                    jutsu_ph.markdown("*Perform a jutsu sequence…*")

                seq = detector.get_sequence()
                if seq:
                    chips = "".join(
                        f'<span class="sign-chip">{s}</span>' +
                        (' <span style="color:#888">→</span> ' if i < len(seq)-1 else "")
                        for i, s in enumerate(seq))
                    seq_ph.markdown(chips, unsafe_allow_html=True)
                else:
                    seq_ph.markdown("*No signs yet.*")

                if st.session_state.jutsu_log:
                    log_ph.markdown("\n".join(
                        f"- {j}" for j in reversed(st.session_state.jutsu_log[-10:])))

                time.sleep(0.03)
        finally:
            cap.release()
else:
    frame_ph.info("Click ▶ Start to begin.")
