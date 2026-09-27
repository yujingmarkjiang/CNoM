from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from cnom.data.manifest import save_manifest
from cnom.data.video import VideoDecoder

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def prepare_manifest(config: dict[str, Any], input_root: str | None = None, output: str | None = None) -> pd.DataFrame:
    adapter = str(config["adapter"])
    root = Path(input_root or config.get("root", ".")).expanduser().resolve()
    handlers: dict[str, Callable[[dict[str, Any], Path], pd.DataFrame]] = {
        "target_annotations": _target_annotations,
        "class_folders": _class_folders,
        "subject_action_folders": _subject_action_folders,
        "generic_json": _generic_json,
        "anomaly_split_lists": _anomaly_split_lists,
        "anomaly_directories": _anomaly_directories,
        "video_qa_json": _video_qa_json,
        "streaming_json": _streaming_json,
    }
    if adapter not in handlers:
        raise ValueError(f"Unknown dataset adapter: {adapter}")
    frame = handlers[adapter](config, root)
    destination = output or config.get("output")
    if destination:
        save_manifest(frame, destination)
    return frame


def _target_annotations(config: dict[str, Any], root: Path) -> pd.DataFrame:
    annotation_path = Path(config["annotations"])
    if not annotation_path.is_absolute():
        annotation_path = root / annotation_path
    annotations = _read_records(annotation_path)
    labels = {str(k): int(v) for k, v in config["labels"].items()}
    columns = {
        "label": config.get("label_column", "class_name"),
        "start": config.get("start_column", "onset_s"),
        "end": config.get("end_column", "offset_s"),
        "source": config.get("source_column", "recording_id"),
        "camera": config.get("camera_column", "camera_id"),
        "scene": config.get("scene_column", "scene_id"),
        "worker": config.get("worker_column", "worker_proxy"),
        "split": config.get("split_column", "split"),
    }
    path_column = config.get("path_column", "video_path")
    rows: list[dict[str, Any]] = []
    for idx, row in annotations.iterrows():
        label_value = row[columns["label"]]
        label = labels.get(str(label_value), int(label_value) if str(label_value).isdigit() else None)
        if label is None:
            raise ValueError(f"Unknown class label: {label_value}")
        path = _absolute_video_path(root, row[path_column])
        source_id = str(row.get(columns["source"], path.stem))
        start_s, end_s = float(row[columns["start"]]), float(row[columns["end"]])
        rows.append({
            "sample_id": str(row.get("sample_id", f"event-{source_id}-{start_s:.3f}-{end_s:.3f}-{label}-{idx}")), "source_id": source_id,
            "video_path": str(path), "start_s": start_s, "end_s": end_s, "label": label,
            "split": str(row.get(columns["split"], "train")), "camera_id": str(row.get(columns["camera"], "")),
            "scene_id": str(row.get(columns["scene"], "")), "worker_proxy": str(row.get(columns["worker"], "")),
            "event_id": str(row.get("event_id", f"{source_id}:{start_s:.3f}:{end_s:.3f}")),
            "cluster_id": str(row.get("cluster_id", source_id)), "timestamp": row.get("timestamp"),
            "visibility": row.get("visibility"), "ambiguity": row.get("ambiguity"),
            "occluded": _as_bool(row.get("occluded", False)), "ambiguous": _as_bool(row.get("ambiguous", row.get("ambiguity", False))),
        })
    events = pd.DataFrame(rows)
    safe = _sample_safe_segments(events, config.get("safe_sampling", {}))
    frame = pd.concat([events, safe], ignore_index=True)
    events_output = config.get("events_output")
    if events_output:
        save_manifest(events, events_output)
    return frame


def _sample_safe_segments(events: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if not config:
        return pd.DataFrame(columns=events.columns)
    decoder = VideoDecoder()
    segment_s = float(config.get("segment_s", 8.0))
    stride_s = float(config.get("stride_s", 12.0))
    margin = float(config.get("exclusion_margin_s", 4.0))
    max_per_source = int(config.get("max_per_source", 50))
    rows: list[dict[str, Any]] = []
    for source_id, group in events.groupby("source_id"):
        path = Path(group.iloc[0].video_path)
        duration = decoder.info(path).duration_s
        occupied = [(max(0.0, float(row.start_s) - margin), min(duration, float(row.end_s) + margin)) for row in group.itertuples()]
        count = 0
        start = 0.0
        while start + segment_s <= duration and count < max_per_source:
            end = start + segment_s
            if not any(max(start, left) < min(end, right) for left, right in occupied):
                base = group.iloc[0]
                rows.append({
                    "sample_id": f"safe-{source_id}-{start:.3f}", "source_id": source_id, "video_path": str(path),
                    "start_s": start, "end_s": end, "label": 0, "split": base.split,
                    "camera_id": base.get("camera_id", ""), "scene_id": base.get("scene_id", ""),
                    "worker_proxy": "", "event_id": "", "cluster_id": source_id, "timestamp": None,
                    "visibility": "", "ambiguity": "", "occluded": False, "ambiguous": False,
                })
                count += 1
            start += stride_s
    return pd.DataFrame(rows)


def _class_folders(config: dict[str, Any], root: Path) -> pd.DataFrame:
    class_map = {str(k): int(v) for k, v in config["class_map"].items()}
    rows: list[dict[str, Any]] = []
    for class_name, label in class_map.items():
        class_root = root / class_name
        for path in _videos(class_root):
            info = VideoDecoder().info(path)
            depth = max(1, int(config.get("group_from_parent_depth", 1)))
            parents = path.relative_to(class_root).parts[:-1]
            group = "/".join(parents[:depth]) if parents else path.stem
            rows.append(_row(path, label, "unassigned", source_id=path.stem, cluster_id=group, end_s=info.duration_s))
    frame = pd.DataFrame(rows)
    return _assign_grouped_random_splits(frame, config)


def _subject_action_folders(config: dict[str, Any], root: Path) -> pd.DataFrame:
    held_out = str(config.get("held_out_subject", ""))
    action_dirs = sorted({path.parent for path in _videos(root)})
    actions = {name: idx for idx, name in enumerate(sorted({path.name for path in action_dirs}))}
    rows: list[dict[str, Any]] = []
    for path in _videos(root):
        subject = next((part for part in path.parts if re.fullmatch(r"Subject\w+", part, re.I)), path.parent.parent.name)
        action = next((part for part in path.parts if re.fullmatch(r"Action\w+", part, re.I)), path.parent.name)
        split = "test" if subject == held_out else "train"
        info = VideoDecoder().info(path)
        rows.append(_row(path, actions[action], split, source_id=path.stem, cluster_id=subject, end_s=info.duration_s, scene_id=action))
    frame = pd.DataFrame(rows)
    if "validation" not in set(frame["split"]):
        train_subjects = sorted(set(frame.loc[frame["split"] == "train", "cluster_id"]))
        if train_subjects:
            validation_subject = train_subjects[-1]
            frame.loc[frame["cluster_id"] == validation_subject, "split"] = "validation"
    return frame


def _generic_json(config: dict[str, Any], root: Path) -> pd.DataFrame:
    with _config_path(root, config["annotations"]).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get(config.get("json_records_key", "samples"), payload) if isinstance(payload, dict) else payload
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        path = _absolute_video_path(root, record[config.get("path_key", "video")])
        info = VideoDecoder().info(path)
        rows.append(_row(
            path, int(record[config.get("label_key", "label")]), str(record.get(config.get("split_key", "split"), "train")),
            source_id=str(record.get("source_id", path.stem)), cluster_id=str(record.get("cluster_id", path.stem)),
            start_s=float(record.get(config.get("start_key", "start_s"), 0.0)),
            end_s=float(record.get(config.get("end_key", "end_s"), info.duration_s)), sample_id=str(record.get("sample_id", f"json-{idx}")),
        ))
    return pd.DataFrame(rows)


def _anomaly_split_lists(config: dict[str, Any], root: Path) -> pd.DataFrame:
    normal_tokens = [value.lower() for value in config.get("normal_tokens", ["normal"])]
    rows: list[dict[str, Any]] = []
    for split, list_key in (("train", "train_list"), ("test", "test_list")):
        list_path = _config_path(root, config[list_key])
        for line in list_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            value = line.strip().split()[0] if line.strip() else ""
            if not value:
                continue
            path = _absolute_video_path(root, value)
            info = VideoDecoder().info(path)
            label = 0 if any(token in value.lower() for token in normal_tokens) else 1
            row = _row(path, label, split, source_id=path.stem, cluster_id=path.parent.name, end_s=info.duration_s)
            rows.append(row)
    frame = pd.DataFrame(rows)
    annotation_path = config.get("annotations")
    if annotation_path and Path(str(annotation_path)).suffix.lower() in {".txt", ".list"}:
        intervals = _read_temporal_annotations(_config_path(root, annotation_path))
        if intervals:
            frame["temporal_annotations"] = frame["video_path"].map(lambda value: json.dumps(intervals.get(Path(value).stem, [])))
    train_groups = sorted(frame.loc[frame["split"] == "train", "cluster_id"].unique())
    validation_groups = set(train_groups[-max(1, len(train_groups) // 10):])
    frame.loc[(frame["split"] == "train") & frame["cluster_id"].isin(validation_groups), "split"] = "validation"
    return frame


def _anomaly_directories(config: dict[str, Any], root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    masks = _mask_map(root, config)
    for path in _glob_videos(root, config.get("train_normal_glob", "train/**/*")):
        info = VideoDecoder().info(path)
        rows.append(_row(path, 0, "train", source_id=path.stem, cluster_id=path.parent.name, end_s=info.duration_s))
    for path in _glob_videos(root, config.get("test_video_glob", "test/**/*")):
        info = VideoDecoder().info(path)
        label = 0 if "normal" in str(path).lower() else 1
        row = _row(path, label, "test", source_id=path.stem, cluster_id=path.parent.name, end_s=info.duration_s)
        if path.stem in masks:
            row["frame_labels_path"] = masks[path.stem]
            label = int(np.asarray(_load_mask(Path(masks[path.stem]))).astype(bool).any())
            row["label"] = label
        rows.append(row)
    frame = pd.DataFrame(rows)
    train_groups = sorted(frame.loc[frame["split"] == "train", "cluster_id"].unique())
    validation_groups = set(train_groups[-max(1, len(train_groups) // 10):])
    frame.loc[(frame["split"] == "train") & frame["cluster_id"].isin(validation_groups), "split"] = "validation"
    return frame


def _video_qa_json(config: dict[str, Any], root: Path) -> pd.DataFrame:
    records = _read_json_records(_config_path(root, config["annotations"]))
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        path = _absolute_video_path(root, record[config.get("video_key", "video")])
        info = VideoDecoder().info(path)
        row = _row(path, int(record.get("label", 0)), str(record.get("split", "test")), source_id=path.stem, cluster_id=path.stem, end_s=info.duration_s, sample_id=str(record.get("id", idx)))
        row.update({"question": record.get(config.get("question_key", "question")), "answer": record.get(config.get("answer_key", "answer")), "category": record.get(config.get("category_key", "category"))})
        rows.append(row)
    return pd.DataFrame(rows)


def _streaming_json(config: dict[str, Any], root: Path) -> pd.DataFrame:
    records = _read_json_records(_config_path(root, config["annotations"]))
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        path = _absolute_video_path(root, record[config.get("video_key", "video")])
        info = VideoDecoder().info(path)
        safe = _as_bool(record.get(config.get("safe_key", "is_safe"), False))
        start = float(record.get(config.get("risk_onset_key", "risk_onset_s"), 0.0))
        end = float(record.get(config.get("risk_end_key", "risk_end_s"), info.duration_s))
        row = _row(path, 0 if safe else 1, str(record.get("split", "test")), source_id=path.stem, cluster_id=path.stem, start_s=0.0 if safe else start, end_s=info.duration_s if safe else end, sample_id=str(record.get("id", idx)))
        row.update({"risk_onset_s": None if safe else start, "risk_end_s": None if safe else end, "domain": record.get(config.get("domain_key", "domain"), "")})
        rows.append(row)
    return pd.DataFrame(rows)


def _assign_grouped_random_splits(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rng = np.random.default_rng(int(config.get("seed", 2027)))
    groups = np.array(sorted(frame["cluster_id"].dropna().astype(str).unique()))
    if len(groups) < 3:
        raise ValueError("Grouped random splitting requires at least three distinct cluster_id values")
    rng.shuffle(groups)
    test_n = max(1, int(round(len(groups) * float(config.get("test_fraction", 0.2)))))
    validation_n = max(1, int(round(len(groups) * float(config.get("validation_fraction", 0.15)))))
    if test_n + validation_n >= len(groups):
        validation_n = 1
        test_n = 1
    test, validation = set(groups[:test_n]), set(groups[test_n:test_n + validation_n])
    frame.loc[frame["cluster_id"].isin(test), "split"] = "test"
    frame.loc[frame["cluster_id"].isin(validation), "split"] = "validation"
    frame.loc[~frame["cluster_id"].isin(test | validation), "split"] = "train"
    return frame


def _read_records(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.DataFrame(_read_json_records(path))
    raise ValueError(f"Unsupported annotation format: {path}")


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("samples", "records", "data", "videos"):
        if key in payload and isinstance(payload[key], list):
            return payload[key]
    raise ValueError(f"Cannot locate a record list in {path}")


def _row(
    path: Path, label: int, split: str, source_id: str, cluster_id: str, end_s: float,
    start_s: float = 0.0, sample_id: str | None = None, scene_id: str = "",
) -> dict[str, Any]:
    return {
        "sample_id": sample_id or f"{source_id}-{start_s:.3f}-{end_s:.3f}", "source_id": source_id,
        "video_path": str(path), "start_s": start_s, "end_s": end_s, "label": int(label), "split": split,
        "camera_id": "", "scene_id": scene_id, "event_id": "", "cluster_id": cluster_id, "timestamp": None,
    }


def _absolute_video_path(root: Path, value: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.is_file():
        return path
    names = [path.name] if path.suffix else [path.name + suffix for suffix in sorted(VIDEO_SUFFIXES)]
    candidates = [candidate for name in names for candidate in root.rglob(name) if candidate.is_file()]
    unique = sorted({candidate.resolve() for candidate in candidates})
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise FileNotFoundError(f"Ambiguous video path {value}; matches: {unique[:5]}")
    raise FileNotFoundError(path)


def _videos(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)


def _glob_videos(root: Path, pattern: str) -> list[Path]:
    paths = {path.resolve() for expanded in _expand_braces(pattern) for path in root.glob(expanded) if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES}
    return sorted(paths)


def _expand_braces(pattern: str) -> list[str]:
    match = re.search(r"\{([^{}]+)\}", pattern)
    if not match:
        return [pattern]
    values = match.group(1).split(",")
    return [result for value in values for result in _expand_braces(pattern[:match.start()] + value + pattern[match.end():])]


def _config_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_temporal_annotations(path: Path) -> dict[str, list[list[int]]]:
    result: dict[str, list[list[int]]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.strip().split()
        if len(fields) < 4:
            continue
        stem = Path(fields[0]).stem
        numeric: list[int] = []
        for value in fields[1:]:
            try:
                numeric.append(int(float(value)))
            except ValueError:
                continue
        intervals = [[numeric[index], numeric[index + 1]] for index in range(0, len(numeric) - 1, 2) if numeric[index] >= 0 and numeric[index + 1] > numeric[index]]
        if intervals:
            result[stem] = intervals
    return result


def _mask_map(root: Path, config: dict[str, Any]) -> dict[str, str]:
    paths: list[Path] = []
    if config.get("test_mask_glob"):
        paths.extend(path for expanded in _expand_braces(str(config["test_mask_glob"])) for path in root.glob(expanded) if path.is_file())
    annotations = config.get("annotations")
    if annotations:
        location = _config_path(root, annotations)
        if location.is_dir():
            paths.extend(path for path in location.rglob("*") if path.is_file() and path.suffix.lower() in {".npy", ".txt", ".csv"})
    return {path.stem.replace("_frame_mask", "").replace("_mask", ""): str(path.resolve()) for path in paths}



def _load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).reshape(-1)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, header=None).to_numpy().reshape(-1)
    values = np.fromstring(path.read_text(encoding="utf-8", errors="ignore").replace(",", " "), sep=" ")
    if values.size == 0:
        raise ValueError(f"Frame-label mask is empty: {path}")
    return values

def _as_bool(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "t"}:
        return True
    if normalized in {"false", "0", "no", "n", "f", "", "none", "nan"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")
