from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    blob = torch.load(args.checkpoint, map_location="cpu")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
