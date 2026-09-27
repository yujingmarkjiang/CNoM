from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frames: int
    duration_s: float
    width: int
    height: int


class VideoDecoder:
    def __init__(self, backend: str = "opencv") -> None:
        if backend != "opencv":
            raise ValueError(f"Unsupported decoder backend: {backend}")
        self.backend = backend

    def info(self, path: str | Path) -> VideoInfo:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        duration = frames / fps if fps > 0 else 0.0
        return VideoInfo(fps, frames, duration, width, height)

    def decode(self, path: str | Path, start_s: float, end_s: float, num_frames: int, sampling: str = "uniform") -> torch.Tensor:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Video metadata is invalid: {path}")
        start_frame = max(0, min(frame_count - 1, int(math.floor(start_s * fps))))
        end_frame = max(start_frame + 1, min(frame_count, int(math.ceil(end_s * fps))))
        indices = self._indices(start_frame, end_frame, num_frames, sampling)
        frames: list[np.ndarray] = []
        last: np.ndarray | None = None
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last = frame
            elif last is None:
                cap.release()
                raise RuntimeError(f"Failed to decode frame {index} from {path}")
            frames.append(last.copy())
        cap.release()
        array = np.stack(frames)
        return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


    def iter_frames(self, path: str | Path, start_s: float = 0.0, end_s: float | None = None, analyzed_fps: float = 10.0):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if source_fps <= 0 or frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Video metadata is invalid: {path}")
        end_s = min(frame_count / source_fps, end_s if end_s is not None else frame_count / source_fps)
        step = max(1, int(round(source_fps / analyzed_fps)))
        index = max(0, int(round(start_s * source_fps)))
        final = min(frame_count, int(round(end_s * source_fps)))
        while index < final:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield index / source_fps, torch.from_numpy(frame).permute(2, 0, 1).contiguous()
            index += step
        cap.release()

    @staticmethod
    def _indices(start: int, end: int, count: int, sampling: str) -> np.ndarray:
        if count <= 0:
            raise ValueError("num_frames must be positive")
        if sampling == "uniform":
            return np.linspace(start, max(start, end - 1), count).round().astype(np.int64)
        if sampling == "random":
            if end - start >= count:
                return np.sort(np.random.choice(np.arange(start, end), count, replace=False))
            return np.sort(np.random.choice(np.arange(start, end), count, replace=True))
        if sampling == "head":
            values = np.arange(start, min(end, start + count))
            return np.pad(values, (0, count - len(values)), mode="edge")
        raise ValueError(f"Unknown sampling mode: {sampling}")


class VideoTransform:
    def __init__(
        self,
        image_size: int = 224,
        train: bool = False,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        horizontal_flip: float = 0.5,
        color_jitter: float = 0.15,
        random_erasing: float = 0.0,
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.horizontal_flip = horizontal_flip
        self.color_jitter = color_jitter
        self.random_erasing = random_erasing

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        video = video.float().div_(255.0)
        video = self._resize_crop(video)
        if self.train and torch.rand(()) < self.horizontal_flip:
            video = video.flip(-1)
        if self.train and self.color_jitter > 0:
            gain = 1 + (torch.rand(3) * 2 - 1) * self.color_jitter
            video = (video * gain.view(1, 3, 1, 1)).clamp(0, 1)
        if self.train and self.random_erasing > 0 and torch.rand(()) < self.random_erasing:
            video = self._erase(video)
        return (video - self.mean) / self.std

    def _resize_crop(self, video: torch.Tensor) -> torch.Tensor:
        _, _, height, width = video.shape
        short = min(height, width)
        scale = self.image_size / short
        target_h, target_w = int(round(height * scale)), int(round(width * scale))
        video = F.interpolate(video, size=(target_h, target_w), mode="bilinear", align_corners=False)
        if self.train:
            top = int(torch.randint(0, max(1, target_h - self.image_size + 1), ()).item())
            left = int(torch.randint(0, max(1, target_w - self.image_size + 1), ()).item())
        else:
            top, left = (target_h - self.image_size) // 2, (target_w - self.image_size) // 2
        return video[:, :, top:top + self.image_size, left:left + self.image_size]

    def _erase(self, video: torch.Tensor) -> torch.Tensor:
        _, _, height, width = video.shape
        erase_h, erase_w = max(1, height // 8), max(1, width // 8)
        top = int(torch.randint(0, max(1, height - erase_h + 1), ()).item())
        left = int(torch.randint(0, max(1, width - erase_w + 1), ()).item())
        video[:, :, top:top + erase_h, left:left + erase_w] = 0
        return video
