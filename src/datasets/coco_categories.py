"""COCO 2017 category definitions and prompt construction.

Provides the standard 80 COCO category names, ID mappings, and utilities
for constructing text prompts compatible with Grounding DINO.

Reference: https://cocodataset.org/#format-data
"""

# COCO 80 category names in order of category_id (1-80).
# This ordering matches the official COCO annotation category IDs.
COCO_80_CLASSES: list[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Official COCO category IDs (non-sequential, IDs 1-90 with gaps).
# These match the actual COCO annotation file category_id values.
_COCO_CATEGORY_IDS: list[int] = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]

# Mapping from COCO category_id to category name
CATEGORY_ID_TO_NAME: dict[int, str] = dict(zip(_COCO_CATEGORY_IDS, COCO_80_CLASSES, strict=True))

# Mapping from category name to COCO category_id
CATEGORY_NAME_TO_ID: dict[str, int] = dict(zip(COCO_80_CLASSES, _COCO_CATEGORY_IDS, strict=True))


def build_coco_prompt(separator: str = " . ") -> str:
    """Build a text prompt containing all 80 COCO categories.

    Args:
        separator: String to place between category names.
            Default is ' . ' as recommended by Grounding DINO.

    Returns:
        Formatted prompt string, e.g., "person . bicycle . car . ..."
    """
    return separator.join(COCO_80_CLASSES) + " ."


def match_phrase_to_category(phrase: str) -> int | None:
    """Match a detected phrase to a COCO category.

    Uses substring matching to handle cases where Grounding DINO
    outputs partial category names or variations.

    Args:
        phrase: Detected phrase from Grounding DINO.

    Returns:
        COCO category_id (1-80) if matched, None otherwise.
    """
    phrase_lower = phrase.lower().strip()

    # Exact match first
    if phrase_lower in CATEGORY_NAME_TO_ID:
        return CATEGORY_NAME_TO_ID[phrase_lower]

    # Check if phrase is a substring of any category name
    for name, cat_id in CATEGORY_NAME_TO_ID.items():
        if phrase_lower in name or name in phrase_lower:
            return cat_id

    # Handle common aliases
    aliases = {
        "tv": "tv",
        "television": "tv",
        "monitor": "tv",
        "cellphone": "cell phone",
        "mobile phone": "cell phone",
        "phone": "cell phone",
        "sofa": "couch",
        "diningtable": "dining table",
        "pottedplant": "potted plant",
        "hairdryer": "hair drier",
        "teddybear": "teddy bear",
        "firehydrant": "fire hydrant",
        "stopsign": "stop sign",
        "parkingmeter": "parking meter",
        "trafficlight": "traffic light",
        "sportsball": "sports ball",
        "baseballbat": "baseball bat",
        "baseballglove": "baseball glove",
        "tennisracket": "tennis racket",
        "wineglass": "wine glass",
        "hotdog": "hot dog",
    }

    for alias, canonical in aliases.items():
        if phrase_lower == alias or phrase_lower in alias:
            return CATEGORY_NAME_TO_ID.get(canonical)

    return None


def build_phrase_to_category_mapping(phrases: list[str]) -> dict[str, int]:
    """Build a mapping from detected phrases to COCO category IDs.

    Args:
        phrases: List of unique phrases detected by the model.

    Returns:
        Dictionary mapping phrase -> category_id.
    """
    mapping = {}
    for phrase in set(phrases):
        cat_id = match_phrase_to_category(phrase)
        if cat_id is not None:
            mapping[phrase] = cat_id
    return mapping
