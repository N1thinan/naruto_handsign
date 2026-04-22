# ─────────────────────────────────────────────────────────────
#  sequence_detector.py  –  Rolling buffer + jutsu matching
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from collections import deque
from utils.constants import (
    JUTSU_DICT,
    SEQUENCE_MAX_LEN,
    HOLD_FRAMES_REQUIRED,
    JUTSU_DISPLAY_FRAMES,
)

# Pre-compute which windows are prefixes of longer jutsu combos
_PREFIX_SET: set[tuple] = set()
for _key in JUTSU_DICT:
    for _i in range(1, len(_key)):
        _PREFIX_SET.add(_key[:_i])


class SequenceDetector:
    """
    Tracks the sequence of confirmed hand signs and checks for jutsu combos.

    Matching rules:
      - Signs must be held stable for HOLD_FRAMES_REQUIRED frames to register.
      - Longest-window match is tried first.
      - If a match is also a prefix of a longer combo, we wait.
        After IDLE_FLUSH_FRAMES of no detected hand, we force-fire the best
        available match (so Clone Jutsu fires even though Ram→Serpent is a
        prefix of the longer Ram→Serpent→Tiger combo).
    """

    IDLE_FLUSH_FRAMES = 20   # frames of no-hand before flushing a pending prefix

    def __init__(self):
        self._sequence:   deque[str] = deque(maxlen=SEQUENCE_MAX_LEN)
        self._hold_sign:  str | None = None
        self._hold_count: int        = 0
        self._idle_frames: int       = 0

        self._jutsu_name:   str | None = None
        self._jutsu_emoji:  str | None = None
        self._jutsu_colour: tuple      = (255, 255, 255)
        self._jutsu_timer:  int        = 0

    # ── Public interface ──────────────────────────────────────

    def update(self, predicted_sign: str | None) -> None:
        """Feed the current frame's predicted sign (or None if no hand)."""
        if self._jutsu_timer > 0:
            self._jutsu_timer -= 1
            if self._jutsu_timer == 0:
                self._clear_jutsu()

        if predicted_sign is None:
            self._idle_frames += 1
            self._reset_hold()
            if self._idle_frames == self.IDLE_FLUSH_FRAMES:
                self._check_jutsu(force=True)
            return

        self._idle_frames = 0

        if predicted_sign == self._hold_sign:
            self._hold_count += 1
        else:
            self._hold_sign  = predicted_sign
            self._hold_count = 1

        if self._hold_count == HOLD_FRAMES_REQUIRED:
            self._register(predicted_sign)
            self._hold_count = HOLD_FRAMES_REQUIRED + 1  # prevent re-fire

    def jutsu_active(self) -> bool:
        return self._jutsu_timer > 0

    def current_jutsu(self) -> tuple[str, str, tuple]:
        return self._jutsu_name, self._jutsu_emoji, self._jutsu_colour

    def get_sequence(self) -> list[str]:
        return list(self._sequence)

    def get_hold_progress(self) -> float:
        if self._hold_sign is None:
            return 0.0
        return min(self._hold_count / HOLD_FRAMES_REQUIRED, 1.0)

    def reset_sequence(self) -> None:
        self._sequence.clear()
        self._reset_hold()
        self._idle_frames = 0

    # ── Internal ──────────────────────────────────────────────

    def _register(self, sign: str) -> None:
        seq = list(self._sequence)
        if len(seq) >= 2 and seq[-1] == sign and seq[-2] == sign:
            return  # block triple+ repeats
        self._sequence.append(sign)
        self._check_jutsu(force=False)

    def _check_jutsu(self, force: bool = False) -> None:
        """
        Match longest suffix of buffer against JUTSU_DICT.
        force=True  → fire even if the window is a prefix of a longer combo.
        force=False → skip prefix windows; wait for more signs.
        """
        seq = list(self._sequence)
        if not seq:
            return

        for length in range(min(len(seq), SEQUENCE_MAX_LEN), 0, -1):
            window = tuple(seq[-length:])
            if window not in JUTSU_DICT:
                continue
            if not force and window in _PREFIX_SET:
                continue  # wait — longer combo still possible
            name, emoji, colour = JUTSU_DICT[window]
            self._activate_jutsu(name, emoji, colour)
            self._sequence.clear()
            self._idle_frames = 0
            return

    def _activate_jutsu(self, name: str, emoji: str, colour: tuple) -> None:
        self._jutsu_name   = name
        self._jutsu_emoji  = emoji
        self._jutsu_colour = colour
        self._jutsu_timer  = JUTSU_DISPLAY_FRAMES

    def _clear_jutsu(self) -> None:
        self._jutsu_name  = None
        self._jutsu_emoji = None
        self._jutsu_timer = 0

    def _reset_hold(self) -> None:
        self._hold_sign  = None
        self._hold_count = 0
