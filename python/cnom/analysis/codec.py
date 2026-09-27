from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from cnom.data.video import VideoDecoder, VideoTransform
from cnom.models.preprocessing import resolve_video_preprocessing
from cnom.models.registry import build_model
from cnom.utils import autocast_dtype, read_table, resolve_device, write_table


def run_codec_study(config: dict[str, Any]) -> pd.DataFrame:
    study = config["codec"]
    manifest = read_table(study["manifest"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer = CodecScorer(study["model"], config) if study.get("model") else None
    rows: list[dict[str, Any]] = []
    for _, sample in manifest.head(int(study.get("max_clips", len(manifest)))).iterrows():
        reference = scorer.score(Path(sample["video_path"]), float(sample.get("start_s", 0)), float(sample.get("end_s", 0))) if scorer else None
        for profile in study["profiles"]:
            rows.append(_transcode(sample, profile, study, output_dir, scorer, reference))
    result = pd.DataFrame(rows)
    write_table(result, output_dir / "codec_study.csv")
    summary_rows = []
    for profile, group in result.groupby("profile"):
        row = {
            "profile": profile, "clips": len(group), "bytes_mean": float(group["bytes"].mean()),
            "compression_ratio_mean": float(group["compression_ratio"].mean()), "psnr_mean": float(group["psnr_db"].mean()),
            "encode_p95_s": float(np.percentile(group["encode_s"], 95)), "decode_p95_s": float(np.percentile(group["decode_s"], 95)),
        }
        if scorer:
            row.update({
                "prediction_agreement": float(group["prediction_match"].mean()), "encoded_accuracy": float(group["encoded_correct"].mean()),
                "mean_true_class_probability_change": float(group["true_class_probability_change"].mean()),
            })
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    write_table(summary, output_dir / "codec_summary.csv")
    return summary


def _transcode(sample: pd.Series, profile: dict[str, Any], study: dict[str, Any], output_dir: Path, scorer: "CodecScorer | None", reference: dict[str, Any] | None) -> dict[str, Any]:
    source = Path(sample["video_path"])
    start = float(sample.get("start_s", 0))
    source_info = VideoDecoder().info(source)
    end = float(sample.get("end_s", source_info.duration_s))
    end = min(max(end, start + 0.001), source_info.duration_s)
    duration = end - start
    destination = output_dir / "encoded" / f"{sample['sample_id']}_{profile['name']}{profile.get('extension', '.mp4')}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if shutil.which("ffmpeg"):
        subprocess.run(_ffmpeg_command(source, destination, start, duration, profile), check=True)
    else:
        _opencv_transcode(source, destination, start, end, profile)
    encode_s = time.perf_counter() - started
    decode_started = time.perf_counter()
    psnr = _sample_psnr(source, destination, start, end, int(study.get("quality_frames", 8)))
    decode_s = time.perf_counter() - decode_started
    source_segment_bytes = source.stat().st_size * duration / max(source_info.duration_s, 1e-9)
    encoded_bytes = destination.stat().st_size
    row = {
        "sample_id": sample["sample_id"], "profile": profile["name"], "bytes": encoded_bytes,
        "source_segment_bytes": source_segment_bytes, "compression_ratio": encoded_bytes / max(source_segment_bytes, 1),
        "psnr_db": psnr, "encode_s": encode_s, "decode_s": decode_s,
    }
    if scorer and reference:
        encoded = scorer.score(destination, 0, VideoDecoder().info(destination).duration_s)
        label = int(sample["label"])
        row.update({
            "reference_prediction": reference["prediction"], "encoded_prediction": encoded["prediction"],
            "prediction_match": int(reference["prediction"] == encoded["prediction"]), "encoded_correct": int(encoded["prediction"] == label),
            "reference_true_class_probability": reference["probabilities"][label], "encoded_true_class_probability": encoded["probabilities"][label],
            "true_class_probability_change": encoded["probabilities"][label] - reference["probabilities"][label],
        })
    return row


def _ffmpeg_command(source: Path, destination: Path, start: float, duration: float, profile: dict[str, Any]) -> list[str]:
    codec = str(profile["codec"])
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-i", str(source), "-t", str(duration),
        "-vf", f"scale={int(profile['width'])}:{int(profile['height'])}", "-c:v", codec,
    ]
    if codec in {"libx264", "libx265"}:
        command.extend(["-preset", str(profile.get("preset", "medium")), "-crf", str(profile.get("crf", 23))])
    elif codec == "libvpx-vp9":
        command.extend(["-crf", str(profile.get("crf", 32)), "-b:v", str(profile.get("bitrate", "0")), "-row-mt", "1"])
    else:
        if profile.get("bitrate"):
            command.extend(["-b:v", str(profile["bitrate"])])
    command.extend(["-an", str(destination)])
    return command


def _opencv_transcode(source: Path, destination: Path, start_s: float, end_s: float, profile: dict[str, Any]) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    end_frame = int(end_s * fps)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*str(profile.get("fourcc", "mp4v"))), fps, (int(profile["width"]), int(profile["height"])))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer for {destination}")
    index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    while index < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(frame, (int(profile["width"]), int(profile["height"]))))
        index += 1
    writer.release()
    cap.release()


def _sample_psnr(source: Path, encoded: Path, start: float, end: float, frames: int) -> float:
    decoder = VideoDecoder()
    original = decoder.decode(source, start, end, frames, "uniform").float()
    info = decoder.info(encoded)
    reconstructed = decoder.decode(encoded, 0, info.duration_s, frames, "uniform").float()
    reconstructed = torch.nn.functional.interpolate(reconstructed, size=original.shape[-2:], mode="bilinear", align_corners=False)
    mse = float(((original - reconstructed) ** 2).mean())
    return 100.0 if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))


class CodecScorer:
    def __init__(self, model_config: dict[str, Any], config: dict[str, Any]) -> None:
        self.model_config = dict(model_config)
        self.device = resolve_device(str(self.model_config.get("device", config.get("device", "auto"))))
        self.precision = str(self.model_config.get("precision", config.get("precision", "fp32")))
        self.preprocessing = resolve_video_preprocessing(self.model_config, config.get("dataset", {}))
        self.model = build_model(self.model_config, checkpoint=self.model_config.get("checkpoint"), strict=False).to(self.device).eval()
        self.decoder = VideoDecoder(str(self.model_config.get("decoder", "opencv")))
        self.transform = VideoTransform(self.preprocessing.image_size, train=False, mean=self.preprocessing.mean, std=self.preprocessing.std)

    @torch.inference_mode()
    def score(self, path: Path, start_s: float, end_s: float) -> dict[str, Any]:
        video = self.decoder.decode(path, start_s, end_s, self.preprocessing.num_frames, "uniform")
        batch = self.transform(video).unsqueeze(0).to(self.device)
        dtype = autocast_dtype(self.precision)
        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=dtype is not None and self.device.type != "cpu"):
            probabilities = self.model(batch).logits.softmax(-1)[0].cpu().numpy()
        return {"prediction": int(probabilities.argmax()), "probabilities": probabilities}
