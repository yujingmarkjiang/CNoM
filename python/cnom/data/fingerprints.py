from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from cnom.data.manifest import load_manifest, save_manifest
from cnom.data.video import VideoDecoder
from cnom.utils import write_table


def fingerprint_manifest(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    section = config.get("fingerprints", {})
    dataset = config["dataset"]
    frame = load_manifest(dataset["manifest"], root=dataset.get("root"), strict_paths=bool(dataset.get("strict_paths", True)))
    sample_frames = int(section.get("sample_frames", 8))
    hashes = []
    for row in frame.itertuples():
        hashes.append(video_fingerprint(row.video_path, float(row.start_s), float(row.end_s), sample_frames))
    frame["video_fingerprint"] = hashes
    output_manifest = section.get("manifest_output", dataset["manifest"])
    save_manifest(frame, output_manifest)
    pairs = find_near_duplicates(frame, float(section.get("maximum_hamming_fraction", 0.08)), int(section.get("neighbors", 5)))
    output_dir = Path(section.get("output_dir", Path(config.get("output_dir", "runs/default")) / "fingerprints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(pairs, output_dir / "near_duplicate_pairs.csv")
    summary = pd.DataFrame([{
        "samples": len(frame), "candidate_pairs": len(pairs),
        "cross_split_pairs": int(pairs["cross_split"].sum()) if len(pairs) else 0,
        "maximum_hamming_fraction": float(section.get("maximum_hamming_fraction", 0.08)),
    }])
    write_table(summary, output_dir / "fingerprint_summary.csv")
    return {"manifest": frame, "near_duplicate_pairs": pairs, "summary": summary}


def video_fingerprint(path: str | Path, start_s: float, end_s: float, sample_frames: int = 8) -> str:
    video = VideoDecoder().decode(path, start_s, end_s, sample_frames, "uniform").numpy()
    bits = []
    for frame in video:
        image = np.transpose(frame, (1, 2, 0))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(resized)[:8, :8]
        threshold = np.median(dct[1:])
        bits.extend((dct > threshold).reshape(-1).tolist())
    packed = np.packbits(np.asarray(bits, dtype=np.uint8))
    return packed.tobytes().hex()


def find_near_duplicates(frame: pd.DataFrame, maximum_hamming_fraction: float = 0.08, neighbors: int = 5) -> pd.DataFrame:
    if "video_fingerprint" not in frame.columns:
        raise ValueError("Manifest lacks video_fingerprint")
    if len(frame) < 2:
        return pd.DataFrame(columns=["sample_id_a", "sample_id_b", "split_a", "split_b", "source_id_a", "source_id_b", "hamming_fraction", "cross_split"])
    vectors = np.stack([np.unpackbits(np.frombuffer(bytes.fromhex(value), dtype=np.uint8)) for value in frame["video_fingerprint"]])
    count = min(max(2, neighbors + 1), len(frame))
    model = NearestNeighbors(n_neighbors=count, metric="hamming", algorithm="brute", n_jobs=-1).fit(vectors)
    distances, indices = model.kneighbors(vectors)
    rows, seen = [], set()
    for left, (candidate_distances, candidate_indices) in enumerate(zip(distances, indices, strict=True)):
        for distance, right in zip(candidate_distances[1:], candidate_indices[1:], strict=True):
            pair = tuple(sorted((int(left), int(right))))
            if pair in seen or float(distance) > maximum_hamming_fraction:
                continue
            seen.add(pair)
            a, b = frame.iloc[pair[0]], frame.iloc[pair[1]]
            rows.append({
                "sample_id_a": a.sample_id, "sample_id_b": b.sample_id, "split_a": a.split, "split_b": b.split,
                "source_id_a": a.source_id, "source_id_b": b.source_id, "hamming_fraction": float(distance),
                "cross_split": str(a.split) != str(b.split),
            })
    return pd.DataFrame(rows, columns=["sample_id_a", "sample_id_b", "split_a", "split_b", "source_id_a", "source_id_b", "hamming_fraction", "cross_split"])
