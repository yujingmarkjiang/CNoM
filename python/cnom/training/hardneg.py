from __future__ import annotations

from pathlib import Path


def accept(candidates: list[dict], max_score: float = 0.8) -> list[dict]:
    out = []
    for row in candidates:
        if float(row.get("teacher_p_safe", 1)) >= max_score and int(row.get("label", 1)) == 0:
            if str(row.get("review", "pending")) == "accepted":
                out.append(row)
    return out


def write_replay(rows: list[dict], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write("sample_id,label\n")
        for row in rows:
            f.write(f"{row['sample_id']},0\n")
