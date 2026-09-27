from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from cnom.models.sentinel import DistillLoss, Sentinel


def _mobilenet():
    from torchvision.models import mobilenet_v3_small
    m = mobilenet_v3_small(weights=None)
    m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, 9)
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="artifacts/sentinel.pt")
    args = p.parse_args()
    cfg = {}
    path = Path(args.config)
    if path.is_file():
        cfg = yaml.safe_load(path.read_text()) or {}
    model = Sentinel(_mobilenet)
    loss_fn = DistillLoss(
        t=float(cfg.get("loss", {}).get("distillation_temperature", 3.0)),
        alpha=float(cfg.get("loss", {}).get("distillation_weight", 0.7)),
        fn_weight=float(cfg.get("loss", {}).get("positive_class_weight", 4.0)),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "loss": str(loss_fn)}, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
