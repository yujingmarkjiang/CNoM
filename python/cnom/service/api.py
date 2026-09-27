from __future__ import annotations

import hashlib
import hmac
import json
import os
from io import BytesIO
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from cnom import __version__
from cnom.config import load_config
from cnom.service.audit import AuditLog
from cnom.service.database import JobStore
from cnom.service.storage import build_storage


class EventMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    candidate_id: str = Field(min_length=1, max_length=256)
    camera_id: str = Field(min_length=1, max_length=128)
    source_id: str = Field(min_length=1, max_length=256)
    zone_id: str | None = Field(default=None, max_length=128)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    trigger_s: float = Field(ge=0)
    edge_max_score: float = Field(ge=0, le=1)
    edge_mean_score: float = Field(ge=0, le=1)
    callback_url: str | None = None

    @field_validator("start_s", "end_s", "trigger_s", "edge_max_score", "edge_mean_score")
    @classmethod
    def finite_number(cls, value: float) -> float:
        import math
        if not math.isfinite(value):
            raise ValueError("Numeric metadata fields must be finite")
        return value

    def validate_interval(self) -> None:
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")
        if not self.start_s <= self.trigger_s <= self.end_s:
            raise ValueError("trigger_s must fall within the clip interval")


def create_app(config: dict[str, Any] | None = None) -> FastAPI:
    if config is None:
        config = load_config(os.environ.get("CNOM_CLOUD_CONFIG", "configs/deployment/cloud.yaml"))
    cloud = config["cloud"]
    store, storage = JobStore(str(cloud["database_url"])), build_storage(cloud)
    audit = AuditLog(cloud.get("audit_log", "artifacts/cloud_audit.jsonl"))
    api_key = os.environ.get(str(cloud.get("api_key_env", "CNOM_API_KEY")), "")
    if not api_key and not bool(cloud.get("allow_unauthenticated", False)):
        raise RuntimeError("Cloud API key is required unless allow_unauthenticated is explicitly enabled")
    app = FastAPI(title="CNoM cloud verifier", version=__version__)
    app.state.config, app.state.store, app.state.storage, app.state.audit = config, store, storage, audit
    app.add_event_handler("shutdown", store.close)

    def authenticate(x_api_key: str = Header(default="")) -> None:
        if api_key and not hmac.compare_digest(x_api_key, api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "jobs": store.counts()}

    @app.post("/v1/events", dependencies=[Depends(authenticate)])
    async def ingest(
        file: UploadFile = File(...), metadata: str = Form(...), x_content_sha256: str = Header(default=""),
        x_idempotency_key: str = Header(default=""),
    ):
        mime = file.content_type or "application/octet-stream"
        if mime not in set(cloud.get("accepted_mime_types", [])):
            raise HTTPException(status_code=415, detail="Unsupported media type")
        maximum = int(cloud.get("max_upload_bytes", 104857600))
        if len(metadata.encode("utf-8")) > int(cloud.get("max_metadata_bytes", 65536)):
            raise HTTPException(status_code=413, detail="Metadata exceeds upload limit")
        payload = await file.read(maximum + 1)
        if not payload:
            raise HTTPException(status_code=400, detail="Clip is empty")
        if len(payload) > maximum:
            raise HTTPException(status_code=413, detail="Clip exceeds upload limit")
        digest = hashlib.sha256(payload).hexdigest()
        if x_content_sha256 and not hmac.compare_digest(digest, x_content_sha256.lower()):
            raise HTTPException(status_code=400, detail="Checksum mismatch")
        try:
            metadata_value = EventMetadata.model_validate_json(metadata)
            metadata_value.validate_interval()
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid metadata: {exc}") from exc
        metadata_dict = metadata_value.model_dump(mode="json")
        idempotency_key = x_idempotency_key.strip() or hashlib.sha256(f"{metadata_value.camera_id}:{metadata_value.candidate_id}:{digest}".encode()).hexdigest()
        if len(idempotency_key) > 128:
            raise HTTPException(status_code=400, detail="Idempotency key exceeds 128 characters")
        object_key = storage.put(BytesIO(payload), file.filename or "event.mp4")
        try:
            job, created = store.create(object_key, {**metadata_dict, "content_sha256": digest, "mime_type": mime}, idempotency_key)
        except Exception:
            storage.delete(object_key)
            raise
        if not created:
            storage.delete(object_key)
        audit.write(
            "event_ingested" if created else "event_duplicate", job_id=job.id, object_key=job.object_key,
            camera_id=metadata_value.camera_id, candidate_id=metadata_value.candidate_id, content_sha256=digest,
        )
        return JSONResponse({"job_id": job.id, "status": job.status, "created": created}, status_code=202 if created else 200)

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authenticate)])
    def job_status(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job.id, "status": job.status, "result": job.result_json, "attempts": job.attempts, "error": job.error}

    @app.post("/v1/health/edge", dependencies=[Depends(authenticate)])
    def edge_health(payload: dict[str, Any]):
        audit.write("edge_heartbeat", **payload)
        return {"accepted": True}

    return app


def run_api(config: dict[str, Any]) -> None:
    import uvicorn
    cloud = config["cloud"]
    uvicorn.run(
        create_app(config), host=str(cloud.get("host", "0.0.0.0")), port=int(cloud.get("port", 8443)),
        log_level=str(cloud.get("log_level", "info")), ssl_certfile=cloud.get("tls_certfile"), ssl_keyfile=cloud.get("tls_keyfile"),
        proxy_headers=bool(cloud.get("proxy_headers", False)), forwarded_allow_ips=str(cloud.get("forwarded_allow_ips", "127.0.0.1")),
    )
