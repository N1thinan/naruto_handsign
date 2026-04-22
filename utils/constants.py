# ─────────────────────────────────────────────────────────────
#  constants.py  –  Labels, jutsu sequences, UI colours
# ─────────────────────────────────────────────────────────────

# 13 classes — must match your dataset folder names exactly (lowercase)
HAND_SIGNS = [
    "bird",
    "boar",
    "dog",
    "dragon",
    "hare",
    "horse",
    "monkey",
    "ox",
    "ram",
    "rat",
    "snake",
    "tiger",
    "zero",      # neutral / no jutsu pose
]

NUM_CLASSES   = len(HAND_SIGNS)
NUM_LANDMARKS = 21
NUM_FEATURES  = NUM_LANDMARKS * 2   # x, y per landmark → 42 features

# ── Dataset paths ─────────────────────────────────────────────
TRAIN_DIR = "data/train"
TEST_DIR  = "data/test"

# ── Jutsu sequences ───────────────────────────────────────────
# Key   : tuple of sign names in order (lowercase, matching HAND_SIGNS)
# Value : (display_name, emoji, colour_BGR)
# Source: Naruto-Jutsus-With-Hand-Signs.pdf
# Sign mapping: Rabbit→hare, Sheep/Ram→ram, Serpent→snake
JUTSU_DICT = {
    # ── Fire Release ──────────────────────────────────────────
    ("snake", "tiger", "monkey", "boar", "horse", "tiger"):              ("Fire: Grand Fireball",       "🔥", (0,   60, 220)),
    ("snake", "dragon", "hare", "tiger"):                                 ("Fire: Dragon Fire",          "🔥", (0,   60, 200)),
    ("snake", "tiger", "dragon", "hare", "tiger"):                        ("Fire: Flame Dragon",         "🔥", (0,   40, 210)),
    ("snake", "tiger", "dog", "ox", "hare", "tiger"):                     ("Fire: Phoenix",              "🔥", (0,   80, 230)),
    ("rat",   "tiger", "dog",   "ox",  "hare",  "tiger"):                 ("Fire: Phoenix Flower",       "🔥", (0,   50, 215)),
    ("rat",   "horse", "dragon", "ox", "tiger"):                          ("Fire: Dragon Flame Missile", "🔥", (0,   70, 205)),
    ("snake", "ram",   "monkey", "boar", "horse", "tiger"):               ("Fire: Flame Blowing",        "🔥", (0,   30, 195)),

    # ── Ice Release ───────────────────────────────────────────
    ("snake", "rat",  "dragon", "monkey", "tiger", "dog"):                ("Ice: Double Black Dragon",   "❄️", (200, 100,   0)),
    ("tiger", "bird", "horse",  "boar",   "dog"):                         ("Ice: One Horned Whale",      "❄️", (210, 120,   0)),
    ("snake", "hare", "dog"):                                             ("Ice: Rouge Nature",          "❄️", (190,  90,   0)),
    ("bird",  "snake", "monkey", "tiger", "dog"):                         ("Ice: Black Dragon Blizzard", "❄️", (220, 110,   0)),

    # ── Earth Release ─────────────────────────────────────────
    ("ram",   "horse", "dragon"):                                         ("Earth: Dragon Missile",      "🪨", ( 30, 120,  60)),

    # ── Water Release ─────────────────────────────────────────
    ("tiger", "snake", "tiger"):                                          ("Water: Water Wall",          "🌊", (200,  60,   0)),
    ("ox",    "tiger", "ox",    "tiger"):                                 ("Water: Water Dragon Bullet", "🌊", (190,  50,   0)),
    ("ram",   "snake", "tiger", "hare", "snake", "dragon", "hare", "bird"): ("Water: Water Prison",     "🌊", (210,  70,   0)),

    # ── Lightning Release ─────────────────────────────────────
    ("ox",    "hare",  "monkey"):                                         ("Chidori",                    "⚡", (0,  220, 220)),
    ("dog",   "ox",    "hare"):                                           ("Chidori Lv3",                "⚡", (0,  180, 200)),
    ("snake", "dragon", "horse", "ram", "bird", "dog", "ox", "hare", "monkey"): ("Chidori Lv2",         "⚡", (0,  200, 240)),

    # ── Ninjutsu ──────────────────────────────────────────────
    ("ram",   "snake", "tiger"):                                          ("Shadow Clone Jutsu",         "💨", (180, 180,   0)),
    ("tiger", "boar",  "ox",  "dog"):                                     ("Clone Jutsu",                "💨", (160, 160,   0)),
    ("dog",   "boar",  "tiger"):                                          ("Transformation Jutsu",       "💨", (150, 150,   0)),
    ("tiger", "boar",  "ox",  "snake"):                                   ("Substitution Jutsu",         "💨", (140, 130,   0)),
    ("tiger", "ox",    "dog"):                                            ("Sakura's Clone Jutsu",       "💨", (170, 170,   0)),

    # ── Genjutsu / Other ──────────────────────────────────────
    ("boar",  "dog",   "monkey", "bird",  "ram"):                         ("Summoning Jutsu",            "🐉", (130,   0, 200)),
    ("tiger", "dog",   "dragon"):                                         ("Reanimation Jutsu",          "💀", ( 50,  50,  50)),
    ("boar",  "dog",   "tiger",  "monkey", "ram"):                        ("Summon: Dog Pack",           "🐕", (100,  50, 150)),
    ("snake", "boar",  "ram",   "hare",   "dog",  "rat", "bird", "horse", "snake"): ("Death God Summon", "💀", ( 30,  30,  70)),
    ("boar",  "dog",   "bird",  "monkey", "ram"):                         ("Gamabunta",                  "🐸", ( 80,   0, 120)),
    ("hare",  "bird",  "snake", "horse",  "monkey", "horse", "ram", "rat"): ("Cursed Seal Jutsu",        "🔱", ( 50,   0, 100)),
    ("dog",   "snake", "bird",  "tiger"):                                 ("Reversal Jutsu",             "🔄", (100, 100, 100)),
    ("dog",   "snake", "ox",   "bird",   "tiger"):                        ("Time Reversal Jutsu",        "⏪", ( 80,  80, 130)),
    ("tiger", "dragon", "rat", "tiger"):                                  ("Tile Shuriken",              "🌟", (100, 180,  80)),
    ("horse", "tiger"):                                                   ("Dispell",                    "✨", (200, 200,   0)),
}

# ── Sequence detection settings ───────────────────────────────
SEQUENCE_MAX_LEN     = 10   # rolling buffer length
HOLD_FRAMES_REQUIRED = 8    # stable frames before a sign is registered
JUTSU_DISPLAY_FRAMES = 90   # ~3 s at 30 fps

# ── UI colours (BGR) ─────────────────────────────────────────
COLOUR_LANDMARK   = (0,   220, 100)
COLOUR_CONNECTION = (0,   160,  60)
COLOUR_HUD_BG     = (20,   20,  20)
COLOUR_WHITE      = (255, 255, 255)
COLOUR_YELLOW     = (0,   220, 220)
COLOUR_GREY       = (160, 160, 160)
