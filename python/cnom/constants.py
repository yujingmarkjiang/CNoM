LABELS = [
    "safe_background",
    "no_fall_protection",
    "leaning_on_machine",
    "near_moving_parts",
    "missing_safety_guard",
    "loose_clothing",
    "under_suspended_load",
    "machine_vibration",
    "unsafe_tool_posture",
]

LABEL_TO_ID = {name: idx for idx, name in enumerate(LABELS)}
ID_TO_LABEL = {idx: name for idx, name in enumerate(LABELS)}
NON_SUPPRESSIBLE = ("near_moving_parts", "under_suspended_load")
NON_SUPPRESSIBLE_IDS = (3, 6)
