from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import torch

from cnom.data.video import VideoTransform
from cnom.pipeline.spool import EncryptedSpool, parse_spool_key
from cnom.pipeline.temporal import CandidateEvent, StreamingTemporalGate, StreamingTemporalPolicy
from cnom.utils import autocast_dtype, resolve_device, sha256_file

LOGGER = logging.getLogger(__name__)


class EdgeAgent:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config["edge"]
        self.agent_id = str(self.config.get("agent_id") or uuid.uuid4())
        self.device = resolve_device(str(self.config.get("device", "auto")))
        self.precision = str(self.config.get("precision", "fp32"))
        api_key_env = str(self.config.get("api_key_env", "CNOM_API_KEY"))
        if not os.environ.get(api_key_env) and not bool(self.config.get("allow_unauthenticated", False)):
            raise RuntimeError(f"Edge upload API key is required in {api_key_env}")
        self.model_path = Path(self.config["model_path"])
        self.model = torch.jit.load(str(self.model_path), map_location=self.device).eval()
        self.model_sha256 = sha256_file(self.model_path)
        metadata = self._model_metadata()
        preprocessing = metadata.get("preprocessing", {})
        image_size = int(self.config.get("image_size", preprocessing.get("image_size", 224)))
        mean = tuple(self.config.get("mean", preprocessing.get("mean", (0.485, 0.456, 0.406))))
        std = tuple(self.config.get("std", preprocessing.get("std", (0.229, 0.224, 0.225))))
        self.transform = VideoTransform(image_size, train=False, mean=mean, std=std)
        self.policy = StreamingTemporalPolicy(
            frame_threshold=float(self.config["frame_threshold"]), window_frames=int(self.config["window_frames"]),
            persistence=float(self.config["persistence"]), pre_roll_s=float(self.config.get("pre_roll_s", 2.0)),
            decision_post_roll_s=float(self.config.get("decision_post_roll_s", 0.4)), max_clip_s=float(self.config.get("decision_max_clip_s", 5.0)),
            reset_persistence=float(self.config.get("reset_persistence", 0.20)), reset_seconds=float(self.config.get("reset_seconds", 2.0)),
            cooldown_seconds=float(self.config.get("cooldown_seconds", 8.0)),
        )
        key_value = os.environ.get(str(self.config.get("encryption_key_env", "CNOM_SPOOL_KEY")), "")
        key = parse_spool_key(key_value) if key_value else None
        self.spool = EncryptedSpool(
            self.config["spool_dir"], self.config["spool_database"], int(self.config["spool_max_bytes"]),
            int(self.config["spool_max_items"]), int(self.config["spool_max_age_s"]), bool(self.config.get("encrypt_spool", True)), key,
        )
        self.stop_event = threading.Event()
        self.last_heartbeat = 0.0


    def _model_metadata(self) -> dict[str, Any]:
        path = Path(self.config.get("model_metadata_path", str(self.model_path) + ".json"))
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def run(self) -> None:
        self._start_metrics_server()
        uploader = threading.Thread(target=self._upload_loop, daemon=True, name="uploader")
        uploader.start()
        workers = [threading.Thread(target=self._stream_loop, args=(source,), daemon=True, name=str(source["id"])) for source in self.config["sources"]]
        for worker in workers:
            worker.start()
        try:
            while not self.stop_event.is_set():
                if not uploader.is_alive():
                    raise RuntimeError("Uploader thread stopped unexpectedly")
                self._heartbeat()
                self.stop_event.wait(float(self.config.get("heartbeat_s", 30)))
        except KeyboardInterrupt:
            LOGGER.info("Stopping edge agent")
        finally:
            self.stop_event.set()
            for worker in workers:
                worker.join(timeout=10)
            uploader.join(timeout=10)

    def _stream_loop(self, source: dict[str, Any]) -> None:
        reconnect_s = float(self.config.get("stream_reconnect_s", 2.0))
        timestamp_offset = 0.0
        while not self.stop_event.is_set():
            try:
                timestamp_offset = self._stream_session(source, timestamp_offset)
            except Exception as exc:
                LOGGER.exception("Stream session failed camera=%s error=%s", source.get("camera_id"), exc)
            self.stop_event.wait(reconnect_s)

    def _stream_session(self, source: dict[str, Any], timestamp_offset: float) -> float:
        capture = cv2.VideoCapture(str(source["uri"]))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open stream {source['id']}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or self.config.get("source_fps_fallback", 25.0))
        analyzed_fps = float(self.config.get("analyzed_fps", 10.0))
        if source_fps <= 0 or analyzed_fps <= 0:
            raise ValueError("Source and analyzed frame rates must be positive")
        step = max(1, int(round(source_fps / analyzed_fps)))
        sampled_fps = source_fps / step
        source_id = f"{source['id']}:{self.agent_id}"
        gate = StreamingTemporalGate(str(source["camera_id"]), source_id, self.policy)
        buffer: deque[tuple[float, np.ndarray]] = deque()
        pending: list[CandidateEvent] = []
        frame_index, failures = 0, 0
        failure_limit = int(self.config.get("stream_read_failure_limit", 20))
        try:
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    failures += 1
                    if failures >= failure_limit:
                        raise RuntimeError(f"Consecutive stream read failures reached {failure_limit}")
                    self.stop_event.wait(0.05)
                    continue
                failures = 0
                timestamp_s = timestamp_offset + frame_index / max(source_fps, 1e-6)
                frame_index += 1
                if frame_index % step != 0:
                    continue
                buffer.append((timestamp_s, frame.copy()))
                retention = self.policy.pre_roll_s + self.policy.max_clip_s + 1.0
                while buffer and timestamp_s - buffer[0][0] > retention:
                    buffer.popleft()
                score = self._score_frame(frame)
                pending.extend(gate.update(timestamp_s, score))
                ready = [event for event in pending if timestamp_s >= event.end_s]
                for event in ready:
                    self._encode_and_enqueue(event, buffer, source, sampled_fps)
                    pending.remove(event)
        finally:
            final_timestamp = timestamp_offset + frame_index / max(source_fps, 1e-6)
            pending.extend(gate.flush(final_timestamp))
            for event in pending:
                try:
                    self._encode_and_enqueue(event, buffer, source, sampled_fps)
                except Exception as exc:
                    LOGGER.warning("Could not flush pending event camera=%s event=%s error=%s", source.get("camera_id"), event.candidate_id, exc)
            capture.release()
        return timestamp_offset + frame_index / max(source_fps, 1e-6)

    @torch.inference_mode()
    def _score_frame(self, frame: np.ndarray) -> float:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        video = self.transform(tensor).unsqueeze(0).to(self.device, non_blocking=True)
        dtype = autocast_dtype(self.precision)
        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=dtype is not None and self.device.type != "cpu"):
            output = self.model(video)
        frame_logits = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else output
        logits = frame_logits[:, 0] if frame_logits.ndim == 3 else frame_logits
        unsafe = logits[..., 1:] if logits.shape[-1] > 1 else logits
        return float(unsafe.sigmoid().amax().item())

    def _encode_and_enqueue(
        self, event: CandidateEvent, buffer: deque[tuple[float, np.ndarray]], source: dict[str, Any], output_fps: float,
    ) -> None:
        frames = [frame for timestamp, frame in buffer if event.start_s <= timestamp <= event.end_s]
        if not frames:
            raise RuntimeError(f"No buffered frames for event {event.candidate_id}")
        width, height = int(self.config.get("output_width", 640)), int(self.config.get("output_height", 360))
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            path = Path(handle.name)
        try:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*str(self.config.get("codec", "mp4v"))), output_fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Video encoder could not open codec={self.config.get('codec', 'mp4v')}")
            try:
                for frame in frames:
                    writer.write(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
            finally:
                writer.release()
            metadata = {
                "candidate_id": event.candidate_id, "camera_id": str(source["camera_id"]), "zone_id": source.get("zone_id"),
                "source_id": event.source_id, "start_s": event.start_s, "end_s": event.end_s, "trigger_s": event.trigger_s,
                "edge_max_score": event.max_score, "edge_mean_score": event.mean_score, "callback_url": source.get("callback_url"),
                "agent_id": self.agent_id, "sentinel_sha256": self.model_sha256, "analyzed_fps": output_fps,
            }
            self.spool.enqueue(path, metadata)
        except Exception:
            self.stop_event.set()
            raise
        finally:
            path.unlink(missing_ok=True)

    def _upload_loop(self) -> None:
        key = os.environ.get(str(self.config.get("api_key_env", "CNOM_API_KEY")), "")
        verify: Any = self.config.get("ca_bundle") or bool(self.config.get("verify_tls", True))
        certificate, client_key = self.config.get("client_certificate"), self.config.get("client_key")
        cert = (certificate, client_key) if certificate and client_key else None
        timeout = httpx.Timeout(float(self.config.get("request_timeout_s", 120)), connect=float(self.config.get("connect_timeout_s", 10)))
        lease_s, max_attempts = int(self.config.get("upload_lease_s", 300)), int(self.config.get("upload_max_attempts", 20))
        with httpx.Client(timeout=timeout, verify=verify, cert=cert, http2=bool(self.config.get("http2", True))) as client:
            while not self.stop_event.is_set():
                item = self.spool.lease(lease_s)
                if item is None:
                    self.stop_event.wait(0.25)
                    continue
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
                    path = self.spool.materialize(item, handle.name)
                try:
                    payload = path.read_bytes()
                    digest = hashlib.sha256(payload).hexdigest()
                    response = client.post(
                        str(self.config["upload_url"]),
                        headers={"X-API-Key": key, "X-Content-SHA256": digest, "X-Idempotency-Key": item.id},
                        files={"file": (f"{item.id}.mp4", payload, "video/mp4")}, data={"metadata": json.dumps(item.metadata)},
                    )
                    response.raise_for_status()
                    self.spool.acknowledge(item.id)
                except Exception as exc:
                    if item.attempts >= max_attempts:
                        self.spool.dead_letter(item.id)
                        LOGGER.error("Upload moved to dead letter item=%s attempts=%d error=%s", item.id, item.attempts, exc)
                    else:
                        delay = min(float(self.config.get("retry_max_s", 60)), float(self.config.get("retry_min_s", 1)) * 2 ** min(item.attempts, 10))
                        self.spool.retry(item.id, delay)
                        LOGGER.warning("Upload failed item=%s delay=%.1f error=%s", item.id, delay, exc)
                finally:
                    path.unlink(missing_ok=True)

    def _heartbeat(self) -> None:
        url = self.config.get("heartbeat_url")
        if not url:
            return
        key = os.environ.get(str(self.config.get("api_key_env", "CNOM_API_KEY")), "")
        payload = {
            "time": time.time(), "agent_id": self.agent_id, "sentinel_sha256": self.model_sha256, "spool": self.spool.stats(),
            "sources": [source["id"] for source in self.config["sources"]],
        }
        verify: Any = self.config.get("ca_bundle") or bool(self.config.get("verify_tls", True))
        try:
            response = httpx.post(url, headers={"X-API-Key": key}, json=payload, timeout=10, verify=verify)
            response.raise_for_status()
            self.last_heartbeat = time.time()
        except Exception as exc:
            LOGGER.warning("Heartbeat failed: %s", exc)

    def _start_metrics_server(self) -> None:
        port = int(self.config.get("health_port", 0))
        if port <= 0:
            return
        try:
            from prometheus_client import Gauge, start_http_server
            spool_items = Gauge("cnom_edge_spool_items", "Number of edge spool records")
            spool_bytes = Gauge("cnom_edge_spool_bytes", "Bytes retained in the edge spool")
            start_http_server(port)

            def update() -> None:
                while not self.stop_event.wait(5):
                    stats = self.spool.stats()
                    spool_items.set(float(stats.get("items", 0)))
                    spool_bytes.set(float(stats.get("bytes", 0)))

            threading.Thread(target=update, daemon=True, name="metrics").start()
        except ImportError:
            LOGGER.warning("Prometheus metrics are disabled because prometheus-client is not installed")
