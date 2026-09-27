from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import pandas as pd

from cnom.data.manifest import load_manifest
from cnom.data.video import VideoDecoder
from cnom.utils import write_table


class OpenAICompatibleVLM:
    def __init__(self, provider: dict[str, Any], request: dict[str, Any]) -> None:
        self.base_url = os.environ[str(provider["base_url_env"])].rstrip("/")
        self.api_key = os.environ[str(provider["api_key_env"])]
        self.model = os.environ[str(provider["model_env"])]
        self.request = request
        self.client = httpx.Client(timeout=float(request.get("timeout_s", 120)))

    def evaluate(self, prompt: str, images: list[str]) -> tuple[dict[str, Any], float, dict[str, Any]]:
        content = [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": image}} for image in images]
        payload = {
            "model": self.model, "messages": [{"role": "user", "content": content}],
            "temperature": float(self.request.get("temperature", 0)), "max_tokens": int(self.request.get("max_tokens", 300)),
        }
        last_error: Exception | None = None
        for attempt in range(int(self.request.get("retries", 3))):
            started = time.perf_counter()
            try:
                response = self.client.post(f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"}, json=payload)
                response.raise_for_status()
                raw = response.json()
                text = raw["choices"][0]["message"]["content"]
                return _parse_json(text), time.perf_counter() - started, raw
            except Exception as exc:
                last_error = exc
                time.sleep(min(8, 2 ** attempt))
        raise RuntimeError(f"VLM request failed: {last_error}")


def run_vlm_diagnostic(config: dict[str, Any]) -> pd.DataFrame:
    manifest = load_manifest(config["manifest"], strict_paths=True)
    prompt = Path(config["prompt_file"]).read_text(encoding="utf-8")
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    decoder = VideoDecoder()
    rows: list[dict[str, Any]] = []
    for provider_cfg in config["providers"]:
        provider = OpenAICompatibleVLM(provider_cfg, config["request"])
        for row in manifest.itertuples():
            cache_path = cache_dir / f"{provider_cfg['name']}_{row.sample_id}.json"
            if cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                parsed, latency = cached["parsed"], cached["latency_s"]
            else:
                video = decoder.decode(row.video_path, float(row.start_s), float(row.end_s), int(config.get("frame_samples", 32)), "uniform")
                images = [_data_url(frame) for frame in video]
                parsed, latency, raw = provider.evaluate(prompt, images)
                cache_path.write_text(json.dumps({"parsed": parsed, "latency_s": latency, "raw": raw}, ensure_ascii=False), encoding="utf-8")
            rows.append({
                "provider": provider_cfg["name"], "model": provider.model, "sample_id": row.sample_id, "label": int(row.label),
                "prediction": _label_to_id(str(parsed.get("class", "uncertain"))), "unsafe": bool(parsed.get("unsafe", False)),
                "confidence": float(parsed.get("confidence", 0)), "latency_s": float(latency), "prompt_hash": prompt_hash,
            })
    result = pd.DataFrame(rows)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(result, output_dir / "vlm_predictions.parquet")
    summary = _summarize(result)
    write_table(summary, output_dir / "vlm_diagnostic.csv")
    return summary


def _data_url(frame) -> str:
    bgr = cv2.cvtColor(frame.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode()


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("VLM response does not contain a JSON object")
        return json.loads(match.group(0))


def _label_to_id(value: str) -> int:
    from cnom.constants import LABEL_TO_ID
    return int(LABEL_TO_ID.get(value, 0))


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import f1_score
    rows: list[dict[str, Any]] = []
    for (provider, model), group in frame.groupby(["provider", "model"]):
        values = {"provider": provider, "model": model, "median_latency_s": float(group["latency_s"].median())}
        class_f1 = []
        for label in range(1, 9):
            score = f1_score(group["label"].eq(label), group["prediction"].eq(label), zero_division=0)
            values[f"class_{label}_f1"] = float(score)
            class_f1.append(score)
        values["macro_average"] = float(np.mean(class_f1))
        rows.append(values)
    return pd.DataFrame(rows)
