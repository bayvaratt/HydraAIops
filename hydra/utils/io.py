import gzip
from pathlib import Path
from typing import Iterable, Optional


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _read_text_lines(path: str) -> Iterable[str]:
    with open(path, "r", errors="replace") as f:
        for line in f:
            yield line.rstrip("
")


def read_gz_lines(path: str) -> Iterable[str]:
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            yield line.rstrip("
")


def infer_delim_from_peek(path: str, n: int = 5) -> str:
    lines = []
    reader = read_gz_lines if path.endswith(".gz") else _read_text_lines
    for line in reader(path):
        if line.strip():
            lines.append(line)
        if len(lines) >= n:
            break
    if not lines:
        return ","

    candidates = [",", "	", " "]
    best = ","
    best_score = -1
    for d in candidates:
        counts = [len(l.split(d)) for l in lines]
        score = (len(set(counts)) == 1) * 10 + (counts[0] > 1) * 5 + counts[0]
        if score > best_score:
            best_score = score
            best = d
    return best


def get_earliest_time_in_gz(path: str, delim: str = ",") -> int:
    tmin = None
    for line in read_gz_lines(path):
        if not line.strip():
            continue
        parts = line.split(delim)
        try:
            t = int(parts[0])
        except Exception:
            continue
        tmin = t if tmin is None else min(tmin, t)
    if tmin is None:
        raise ValueError(f"Could not parse timestamps in {path}")
    return tmin


def extract_lanl_window(
    gz_path: str,
    out_path: str,
    t0: int,
    t1: int,
    delim: str = ",",
    max_lines: Optional[int] = None,
) -> int:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with gzip.open(gz_path, "rt", errors="replace") as fin, open(out_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            parts = line.strip().split(delim)
            try:
                t = int(parts[0])
            except Exception:
                continue
            if t0 <= t <= t1:
                fout.write(line)
                written += 1
                if max_lines and written >= max_lines:
                    break
    return written
