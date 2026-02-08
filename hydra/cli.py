import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


def timestamped_out(base_out: str, head: str) -> tuple[str, str]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(base_out)
    if not base.is_absolute():
        base = (REPO_ROOT / base).resolve()
    expected_root = (REPO_ROOT / "runs" / head).resolve()
    if not base.is_relative_to(expected_root):
        print(f"Base output outside {expected_root}, using {expected_root} instead.")
        base = expected_root
    run_id = stamp
    out_dir = base / run_id
    suffix = 1
    while out_dir.exists():
        run_id = f"{stamp}_{suffix:02d}"
        out_dir = base / run_id
        suffix += 1
    out_dir.mkdir(parents=True)
    print(f"Output directory ({head}): {out_dir}")
    return str(out_dir), run_id


def write_run_metadata(
    out_dir: str,
    head: str,
    timestamp: str,
    input_paths: dict,
    row_cap: Optional[int],
    line_cap: Optional[int],
    scan_only: bool,
) -> None:
    payload = {
        "head": head,
        "timestamp": timestamp,
        "out_dir": out_dir,
        "input_paths": input_paths,
        "row_cap": row_cap,
        "line_cap": line_cap,
        "scan_only": scan_only,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "run.json", "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    default_ton = REPO_ROOT / "data/raw/ton_iot_network.csv"
    default_hdfs = REPO_ROOT / "data/raw/HDFS_1"
    default_auth = REPO_ROOT / "data/raw/auth.txt.gz"
    default_red = REPO_ROOT / "data/raw/redteam.txt.gz"

    ap_ton = sub.add_parser("ton", help="Train TON_IoT Statistical Head")
    ap_ton.add_argument("--csv", default=str(default_ton), help="Path to TON_IoT network CSV")
    ap_ton.add_argument("--out", default=str(REPO_ROOT / "runs/ton"), help="Base output directory (timestamp subfolder added)")
    ap_ton.add_argument("--seed", type=int, default=42)
    ap_ton.add_argument("--max_rows", type=int, default=None, help="Optional cap for reading rows")

    ap_hdfs = sub.add_parser("hdfs", help="Train HDFS Semantic Head")
    ap_hdfs.add_argument("--hdfs_dir", default=str(default_hdfs), help="Directory containing HDFS.log and anomaly_label.csv")
    ap_hdfs.add_argument("--out", default=str(REPO_ROOT / "runs/hdfs"), help="Base output directory (timestamp subfolder added)")
    ap_hdfs.add_argument("--seed", type=int, default=42)
    ap_hdfs.add_argument("--max_lines", type=int, default=None, help="Optional cap for reading HDFS.log")
    ap_hdfs.add_argument("--epochs", type=int, default=5, help="Training epochs for the transformer")

    ap_lanl = sub.add_parser("lanl", help="Train LANL Predictive Head")
    ap_lanl.add_argument("--auth_gz", default=str(default_auth), help="Path to auth.txt.gz")
    ap_lanl.add_argument("--redteam_gz", default=str(default_red), help="Path to redteam.txt.gz")
    ap_lanl.add_argument("--out", default=str(REPO_ROOT / "runs/lanl"), help="Base output directory (timestamp subfolder added)")
    ap_lanl.add_argument("--seed", type=int, default=42)
    ap_lanl.add_argument("--window_seconds", type=int, default=7200, help="Extraction window after earliest redteam time")
    ap_lanl.add_argument("--tol_seconds", type=int, default=0, help="Time tolerance when joining auth to redteam")
    ap_lanl.add_argument("--max_auth_lines", type=int, default=2000000, help="Max auth lines to extract in window")
    ap_lanl.add_argument("--scan_only", action="store_true", help="Only scan/extract window (skip training)")

    args = ap.parse_args()

    if args.cmd == "ton":
        from hydra.heads.ton_head import load_ton, train_ton_multiclass

        csv_path = resolve_path(args.csv)
        out_dir, stamp = timestamped_out(args.out, "ton")
        write_run_metadata(
            out_dir=out_dir,
            head="ton",
            timestamp=stamp,
            input_paths={"csv": csv_path},
            row_cap=args.max_rows,
            line_cap=None,
            scan_only=False,
        )
        df = load_ton(args.csv, max_rows=None)  # full file
        # optional cap for smoke
        if args.max_rows:
            df = df.sample(n=min(args.max_rows, len(df)), random_state=args.seed).reset_index(drop=True)
        train_ton_multiclass(df, out_dir=out_dir, seed=args.seed)



    elif args.cmd == "hdfs":
        from hydra.heads.hdfs_head import train_hdfs_semantic

        hdfs_dir = resolve_path(args.hdfs_dir)
        out_dir, stamp = timestamped_out(args.out, "hdfs")
        write_run_metadata(
            out_dir=out_dir,
            head="hdfs",
            timestamp=stamp,
            input_paths={"hdfs_dir": hdfs_dir},
            row_cap=None,
            line_cap=args.max_lines,
            scan_only=False,
        )
        train_hdfs_semantic(
            hdfs_dir,
            out_dir=out_dir,
            seed=args.seed,
            max_lines=args.max_lines,
            epochs=args.epochs,
        )

    elif args.cmd == "lanl":
        from hydra.heads.lanl_head import scan_lanl_window, train_lanl_predictive

        auth_gz = resolve_path(args.auth_gz)
        redteam_gz = resolve_path(args.redteam_gz)
        out_dir, stamp = timestamped_out(args.out, "lanl")
        write_run_metadata(
            out_dir=out_dir,
            head="lanl",
            timestamp=stamp,
            input_paths={"auth_gz": auth_gz, "redteam_gz": redteam_gz},
            row_cap=None,
            line_cap=args.max_auth_lines,
            scan_only=args.scan_only,
        )
        if args.scan_only:
            scan_lanl_window(
                auth_gz=auth_gz,
                redteam_gz=redteam_gz,
                out_dir=out_dir,
                window_seconds=args.window_seconds,
                max_auth_lines=args.max_auth_lines,
            )
            return
        train_lanl_predictive(
            auth_gz=auth_gz,
            redteam_gz=redteam_gz,
            out_dir=out_dir,
            window_seconds=args.window_seconds,
            tol_seconds=args.tol_seconds,
            max_auth_lines=args.max_auth_lines,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
