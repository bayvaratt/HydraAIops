#!/usr/bin/env python3
"""Watchdog for overnight_run.py — checks every 10 minutes, restarts on crash."""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PYTHON = "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
SCRIPT = Path(__file__).resolve().parent / "overnight_run.py"
PROJECT = SCRIPT.parent.parent
LOG = PROJECT / "run_log.txt"
STDOUT_LOG = PROJECT / "overnight_stdout.log"
CHECK_INTERVAL = 600  # 10 minutes
MAX_RESTARTS = 5


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [WATCHDOG] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def is_alive(pid: int) -> bool:
    try:
        result = subprocess.run(["ps", "-p", str(pid)], capture_output=True)
        return result.returncode == 0
    except Exception:
        return False


def get_tail(path: Path, n: int = 30) -> str:
    try:
        result = subprocess.run(
            ["tail", f"-{n}", str(path)], capture_output=True, text=True
        )
        return result.stdout
    except Exception:
        return "(could not read log)"


def has_error(tail: str) -> bool:
    keywords = ["Traceback", "Error", "FATAL", "Exception", "FAILED"]
    return any(k in tail for k in keywords)


def start_script() -> int:
    proc = subprocess.Popen(
        [PYTHON, str(SCRIPT)],
        stdout=open(STDOUT_LOG, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc.pid


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {PYTHON} {__file__} <PID>")
        sys.exit(1)

    pid = int(sys.argv[1])
    restarts = 0

    log(f"Watchdog started — monitoring PID {pid}")

    while True:
        if is_alive(pid):
            tail = get_tail(LOG)
            if has_error(tail):
                log(f"PID {pid} alive but errors in log — check manually")
            else:
                log(f"PID {pid} alive — OK")
        else:
            log(f"PID {pid} is DEAD")
            # Check why
            stdout_tail = get_tail(STDOUT_LOG, 50)
            log_tail = get_tail(LOG)

            if has_error(stdout_tail) or has_error(log_tail):
                log(f"Error detected in logs:\n{stdout_tail[-500:]}")
            else:
                log("No traceback found — process may have been killed externally")

            if restarts >= MAX_RESTARTS:
                log(f"Hit max restarts ({MAX_RESTARTS}). Giving up.")
                break

            # Restart
            pid = start_script()
            restarts += 1
            log(f"Restarted as PID {pid} (restart #{restarts}/{MAX_RESTARTS})")

        time.sleep(CHECK_INTERVAL)

    log("Watchdog exiting.")


if __name__ == "__main__":
    main()
