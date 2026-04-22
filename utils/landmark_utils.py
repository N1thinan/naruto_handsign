# ─────────────────────────────────────────────────────────────
#  landmark_utils.py  –  MediaPipe extraction & normalization
#
#  Uses the MediaPipe Tasks API (mediapipe >= 0.10).
#
#  The hand_landmarker.task model file is downloaded automatically
#  on first run to:  <project_root>/models/hand_landmarker.task
#
#  If you are offline, download it manually from:
#  https://storage.googleapis.com/mediapipe-models/hand_landmarker/
#      hand_landmarker/float16/1/hand_landmarker.task
#  and place it at:  models/hand_landmarker.task
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python            import vision, BaseOptions
from mediapipe.tasks.python.vision     import HandLandmarker, HandLandmarkerOptions, RunningMode

# ── Model file auto-download ──────────────────────────────────
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"


def _ensure_model():
    if _MODEL_PATH.exists():
        return
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {_MODEL_PATH} …")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("  Done.")
    except Exception as e:
        raise RuntimeError(
            f"Could not download hand_landmarker.task: {e}\n"
            f"Download it manually from:\n  {_MODEL_URL}\n"
            f"and place it at:\n  {_MODEL_PATH}"
        ) from e


# ── HAND_CONNECTIONS for drawing ──────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17),             # palm
]


# ── Wrapper that mimics the old mp.solutions.hands interface ──
class _HandsWrapper:
    """
    Thin wrapper around HandLandmarker that exposes a .process(rgb) method
    and a .multi_hand_landmarks attribute so the rest of the code stays
    identical to the legacy API.
    """

    class _LMPoint:
        def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z

    class _HandLM:
        def __init__(self, pts): self.landmark = pts

    def __init__(self, static_image_mode, max_num_hands,
                 min_detection_confidence, min_tracking_confidence):
        _ensure_model()
        mode = RunningMode.IMAGE if static_image_mode else RunningMode.VIDEO
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
            running_mode=mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker  = HandLandmarker.create_from_options(opts)
        self._static_mode = static_image_mode
        self._ts          = 0         # monotonic timestamp for VIDEO mode
        self.multi_hand_landmarks = None

    def process(self, rgb_frame: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        if self._static_mode:
            result = self._landmarker.detect(mp_image)
        else:
            self._ts += 33            # ~30 fps
            result = self._landmarker.detect_for_video(mp_image, self._ts)

        if result.hand_landmarks:
            self.multi_hand_landmarks = [
                self._HandLM([
                    self._LMPoint(lm.x, lm.y, lm.z) for lm in hand
                ])
                for hand in result.hand_landmarks
            ]
        else:
            self.multi_hand_landmarks = None
        return self

    def __enter__(self): return self
    def __exit__(self, *_): self._landmarker.close()
    def close(self): self._landmarker.close()


def get_hands_solution(
    static_image_mode: bool = False,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.7,
    min_tracking_confidence: float = 0.6,
) -> _HandsWrapper:
    """Return a configured hands detector (context manager)."""
    return _HandsWrapper(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


# ── Feature extraction ────────────────────────────────────────

def _landmarks_to_array(hand_lm) -> np.ndarray:
    """Convert a hand landmark object to a normalised (42,) float32 array."""
    pts = np.array([[lm.x, lm.y] for lm in hand_lm.landmark], dtype=np.float32)
    pts -= pts[0]                        # centre on wrist
    scale = np.abs(pts).max()
    if scale > 0:
        pts /= scale
    return pts.flatten()


def extract_landmarks(results) -> np.ndarray | None:
    """
    Given a hands.process() result, return a (42,) normalised float32 array
    or None if no hand was detected.
    """
    if not results.multi_hand_landmarks:
        return None
    return _landmarks_to_array(results.multi_hand_landmarks[0])


def extract_landmarks_from_image(image_rgb: np.ndarray, hands) -> np.ndarray | None:
    """Process a single RGB image and return the feature vector."""
    results = hands.process(image_rgb)
    return extract_landmarks(results)


# ── Drawing helpers ───────────────────────────────────────────

def draw_landmarks(frame, results, colour_landmark, colour_connection):
    """Draw hand skeleton on *frame* (BGR) in-place."""
    if not results.multi_hand_landmarks:
        return
    h, w = frame.shape[:2]
    for hand_lm in results.multi_hand_landmarks:
        pts = hand_lm.landmark
        # Draw connections
        for a, b in HAND_CONNECTIONS:
            x1, y1 = int(pts[a].x * w), int(pts[a].y * h)
            x2, y2 = int(pts[b].x * w), int(pts[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), colour_connection, 2, cv2.LINE_AA)
        # Draw joints
        for lm in pts:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, colour_landmark, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 0), 1,  cv2.LINE_AA)


def draw_bounding_box(frame, results):
    """Draw a bounding rect around the detected hand."""
    if not results.multi_hand_landmarks:
        return
    h, w = frame.shape[:2]
    for hand_lm in results.multi_hand_landmarks:
        xs = [lm.x * w for lm in hand_lm.landmark]
        ys = [lm.y * h for lm in hand_lm.landmark]
        pad = 20
        x1 = max(0, int(min(xs)) - pad)
        y1 = max(0, int(min(ys)) - pad)
        x2 = min(w, int(max(xs)) + pad)
        y2 = min(h, int(max(ys)) + pad)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 180), 1)
