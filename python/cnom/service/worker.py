from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
import torch

from cnom.data.video import VideoDecoder, VideoTransform
from cnom.models.grounding import CocoGrounder
from cnom.models.preprocessing import resolve_video_preprocessing
from cnom.models.registry import build_model
from cnom.pipeline.routing import Router, RouterConfig, resolve_class_ids
from cnom.service.audit import AuditLog
from cnom.service.database import Job, JobStore
from cnom.service.storage import LocalStorage, S3Storage, build_storage
from cnom.utils import autocast_dtype, resolve_device

LOGGER = logging.getLogger(__name__)


class CloudWorker:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        cloud = config["cloud"]
        teacher = dict(cloud["teacher"])
        grounding = dict(config.get("grounding", {}))
        grounding.update(dict(teacher.get("grounding", {})))
        if grounding.get("enabled"):
            teacher["grounding"] = grounding
        self.store, self.storage = JobStore(str(cloud["database_url"])), build_storage(cloud)
        self.audit = AuditLog(cloud.get("audit_log", "artifacts/cloud_audit.jsonl"))
        self.device = resolve_device(str(teacher.get("device", "auto")))
        self.precision, self.temperature = str(teacher.get("precision", "fp32")), float(teacher.get("temperature", 1.34))
        self.model = build_model(teacher, checkpoint=teacher.get("checkpoint"), strict=False).to(self.device).eval()
        self.decoder = VideoDecoder(str(teacher.get("decoder", "opencv")))
        preprocessing = resolve_video_preprocessing(teacher, config.get("dataset", {}))
        self.transform = VideoTransform(preprocessing.image_size, train=False, mean=preprocessing.mean, std=preprocessing.std)
        self.num_frames, self.batch_size = preprocessing.num_frames, int(teacher.get("batch_size", 8))
        self.grounder = CocoGrounder(float(grounding.get("score_threshold", 0.35)), tuple(grounding.get("categories", ()))) if grounding.get("enabled") else None
        routing = cloud["routing"]
        class_ids = resolve_class_ids(routing.get("non_suppressible_classes", ()), config.get("labels", ()))
        self.router = Router(RouterConfig(float(routing["safe_threshold"]), float(routing["unsafe_threshold"]), class_ids))
        self.poll_s, self.lease_s = float(cloud.get("worker_poll_s", 0.25)), int(cloud.get("job_lease_s", 300))
        self.max_attempts, self.callback_timeout_s = int(cloud.get("job_max_attempts", 5)), float(cloud.get("callback_timeout_s", 20))
        self.callback_allowlist = tuple(str(value).lower() for value in cloud.get("callback_allowlist", ()))
        self.callback_token = os.environ.get(str(cloud.get("callback_token_env", "CNOM_CALLBACK_TOKEN")), "")

    def run(self) -> None:
        try:
            while True:
                jobs = self.store.lease_batch(self.batch_size, self.lease_s, self.max_attempts)
                if not jobs:
                    time.sleep(self.poll_s)
                    continue
                self._process_jobs(jobs)
        finally:
            self.store.close()

    def _process_jobs(self, jobs: list[Job]) -> None:
        decoded: list[tuple[Job, Path, Any, torch.Tensor, torch.Tensor | None]] = []
        for job in jobs:
            cleanup = lambda: None
            try:
                path, cleanup = self._materialize(job.object_key)
                raw = self._decode(path)
                transformed = self.transform(raw)
                object_features = self.grounder.extract(raw.float().div(255.0), self.device) if self.grounder else None
                decoded.append((job, path, cleanup, transformed, object_features))
            except Exception as exc:
                cleanup()
                self._fail(job, exc)
        if not decoded:
            return
        try:
            outputs = self._infer([item[3] for item in decoded], [item[4] for item in decoded])
        except Exception as exc:
            for job, _, _, _, _ in decoded:
                self._fail(job, exc)
            LOGGER.exception("Cloud inference batch failed")
        else:
            for (job, _, _, _, _), result in zip(decoded, outputs, strict=True):
                try:
                    self.store.complete(job.id, result)
                    self.audit.write("job_completed", job_id=job.id, result=result)
                except Exception as exc:
                    self._fail(job, exc)
                    continue
                try:
                    self._callback(job.metadata_json.get("callback_url"), job.id, result)
                except Exception as exc:
                    self.audit.write("callback_failed", job_id=job.id, error=str(exc))
                    LOGGER.warning("Callback failed job=%s error=%s", job.id, exc)
        finally:
            for _, _, cleanup, _, _ in decoded:
                cleanup()

    def _decode(self, path: Path) -> torch.Tensor:
        info = self.decoder.info(path)
        if info.duration_s <= 0:
            raise RuntimeError(f"Decoded clip has no duration: {path}")
        return self.decoder.decode(path, 0, info.duration_s, self.num_frames, "uniform")

    @torch.inference_mode()
    def _infer(self, videos: list[torch.Tensor], object_features: list[torch.Tensor | None]) -> list[dict[str, Any]]:
        batch = torch.stack(videos).to(self.device, non_blocking=True)
        features = None
        if self.grounder:
            if len(object_features) != len(videos) or any(value is None for value in object_features):
                raise ValueError("Grounded inference requires one object feature vector per video")
            features = torch.stack([value for value in object_features if value is not None]).to(self.device, non_blocking=True)
        dtype = autocast_dtype(self.precision)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=dtype is not None and self.device.type != "cpu"):
            output = self.model(batch, object_features=features)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - started) / len(videos)
        probabilities = (output.logits / max(self.temperature, 1e-6)).softmax(-1).cpu().numpy()
        results: list[dict[str, Any]] = []
        labels = list(self.config.get("labels", ()))
        for values in probabilities:
            routed = self.router.route(values)
            results.append({
                "route": routed.decision.value, "label": routed.label,
                "label_name": labels[routed.label] if routed.label < len(labels) else str(routed.label),
                "confidence": routed.confidence, "safe_probability": routed.safe_probability,
                "unsafe_probability": routed.unsafe_probability, "probabilities": values.tolist(), "teacher_latency_s": latency,
            })
        return results

    def _fail(self, job: Job, error: Exception) -> None:
        terminal = job.attempts >= self.max_attempts
        self.store.fail(job.id, str(error), terminal=terminal)
        self.audit.write("job_failed", job_id=job.id, error=str(error), terminal=terminal)
        LOGGER.error("Cloud job failed job=%s terminal=%s error=%s", job.id, terminal, error)

    def _materialize(self, object_key: str):
        if isinstance(self.storage, LocalStorage):
            return self.storage.path(object_key), lambda: None
        if isinstance(self.storage, S3Storage):
            temp = tempfile.NamedTemporaryFile(suffix=Path(object_key).suffix, delete=False)
            temp.close()
            path = self.storage.materialize(object_key, temp.name)
            return path, lambda: path.unlink(missing_ok=True)
        raise TypeError("Unsupported storage object")

    def _callback(self, url: str | None, job_id: str, result: dict[str, Any]) -> None:
        if not url:
            return
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not any(host == allowed or host.endswith(f".{allowed}") for allowed in self.callback_allowlist):
            self.audit.write("callback_rejected", job_id=job_id, callback_url=url)
            return
        headers = {"Authorization": f"Bearer {self.callback_token}"} if self.callback_token else {}
        with httpx.Client(timeout=self.callback_timeout_s) as client:
            response = client.post(url, json={"job_id": job_id, "result": result}, headers=headers)
            response.raise_for_status()
