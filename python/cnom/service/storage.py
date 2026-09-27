from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import BinaryIO, Protocol


class ObjectStorage(Protocol):
    def put(self, source: BinaryIO, filename: str) -> str: ...
    def open(self, key: str, mode: str = "rb"): ...
    def delete(self, key: str) -> None: ...


class LocalStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, source: BinaryIO, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        key = f"{uuid.uuid4().hex}{suffix}"
        destination = self.root / key
        with destination.open("wb") as handle:
            shutil.copyfileobj(source, handle)
        return key

    def open(self, key: str, mode: str = "rb"):
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise ValueError("Invalid object key")
        return path.open(mode)

    def path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise ValueError("Invalid object key")
        return path

    def delete(self, key: str) -> None:
        self.path(key).unlink(missing_ok=True)


class S3Storage:
    def __init__(self, bucket: str, prefix: str = "cnom", endpoint_url: str | None = None, region_name: str | None = None, server_side_encryption: str | None = None) -> None:
        import boto3
        self.client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region_name)
        self.bucket, self.prefix, self.server_side_encryption = bucket, prefix.strip("/"), server_side_encryption

    def put(self, source: BinaryIO, filename: str) -> str:
        key = f"{self.prefix}/{uuid.uuid4().hex}{Path(filename).suffix.lower()}"
        arguments = {"ServerSideEncryption": self.server_side_encryption} if self.server_side_encryption else None
        self.client.upload_fileobj(source, self.bucket, key, ExtraArgs=arguments or {})
        return key

    def open(self, key: str, mode: str = "rb"):
        raise RuntimeError("S3 objects must be downloaded through materialize")

    def materialize(self, key: str, destination: str | Path) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(path))
        return path

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)


def build_storage(config: dict):
    backend = str(config.get("storage_backend", "local"))
    if backend == "local":
        return LocalStorage(config.get("storage_root", "artifacts/cloud_clips"))
    if backend == "s3":
        if not config.get("s3_bucket"):
            raise ValueError("s3_bucket is required for S3 storage")
        return S3Storage(config["s3_bucket"], config.get("s3_prefix", "cnom"), config.get("s3_endpoint_url"), config.get("s3_region"), config.get("s3_server_side_encryption"))
    raise ValueError(f"Unknown storage backend: {backend}")
