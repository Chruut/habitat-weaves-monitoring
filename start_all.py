"""Unified launcher — starts all HabitatWeaves servers and opens Chromium in kiosk mode.

Run with:  uv run python start_all.py
"""

import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
os.chdir(BASE)

for d in ["logs", "video_db", "data/audio", "media/audio"]:
    (BASE / d).mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE / "logs"
LOG_FILE = LOG_DIR / "start_all.log"

UV = "/home/loxodongle/.local/bin/uv"
BROWSER = "/usr/bin/chromium"
MAIN_URL = "http://localhost:8080/site_viewer.html"

SERVICES = [
    {
        "name": "static-site",
        "cmd": [UV, "run", "python", "serve_site.py"],
        "ready_port": 8080,
    },
    {
        "name": "sandarium-dashboard",
        "cmd": [
            UV, "run", "panel", "serve", "sandarium_dashboard.py",
            "--port", "5006",
            "--allow-websocket-origin", "*",
        ],
        "ready_port": 5006,
    },
    {
        "name": "camera-player",
        "cmd": [
            UV, "run", "panel", "serve", "camera_player.py",
            "--port", "5007",
            "--static-dirs", "video_db=./video_db",
            "--allow-websocket-origin", "*",
        ],
        "ready_port": 5007,
    },
    {
        "name": "audio-player",
        "cmd": [
            UV, "run", "panel", "serve", "audio_player.py",
            "--port", "5008",
            "--static-dirs", "audio_db=./data/audio", "media_audio=./media/audio",
            "--allow-websocket-origin", "*",
        ],
        "ready_port": 5008,
    },
]

processes: list[subprocess.Popen] = []
browser_proc: subprocess.Popen | None = None


def log(msg: str):
    line = f"[start_all] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def wait_for_port(port: int, timeout: float = 60.0) -> bool:
    """Block until a TCP port accepts connections or timeout expires."""
    import socket
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def shutdown_all(*_args):
    log("Shutting down all services...")
    global browser_proc
    if browser_proc and browser_proc.poll() is None:
        browser_proc.terminate()
        browser_proc = None

    for proc in reversed(processes):
        if proc.poll() is None:
            proc.terminate()

    deadline = time.monotonic() + 8
    for proc in processes:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            log(f"Force-killing pid {proc.pid}")
            proc.kill()

    log("All services stopped.")


def main():
    global browser_proc

    log(f"Starting HabitatWeaves from {BASE}")

    logfile_handle = open(LOG_FILE, "a")

    for svc in SERVICES:
        log(f"Starting {svc['name']}: {' '.join(svc['cmd'])}")
        proc = subprocess.Popen(
            svc["cmd"],
            cwd=str(BASE),
            stdout=logfile_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append(proc)
        log(f"  -> pid {proc.pid}")

    atexit.register(shutdown_all)
    signal.signal(signal.SIGTERM, lambda *a: (shutdown_all(), sys.exit(0)))
    signal.signal(signal.SIGINT, lambda *a: (shutdown_all(), sys.exit(0)))

    log("Waiting for all services to be ready...")
    all_ready = True
    for svc in SERVICES:
        port = svc["ready_port"]
        log(f"  Waiting for {svc['name']} on port {port}...")
        if wait_for_port(port, timeout=90):
            log(f"  {svc['name']} ready.")
        else:
            log(f"  WARNING: {svc['name']} not responding on port {port} after timeout.")
            all_ready = False

    if all_ready:
        log("All services ready.")
    else:
        log("Some services may not be ready — launching browser anyway.")

    kiosk_profile = str(BASE / ".chromium-kiosk")
    browser_args = [
        BROWSER,
        "--kiosk",
        "--noerrdialogs",
        "--disable-infobars",
        "--disable-session-crashed-bubble",
        "--password-store=basic",
        f"--user-data-dir={kiosk_profile}",
        "--no-first-run",
        "--disable-translate",
        "--disable-features=TranslateUI",
        "--autoplay-policy=no-user-gesture-required",
        MAIN_URL,
    ]
    log(f"Opening browser: {' '.join(browser_args)}")
    browser_proc = subprocess.Popen(
        browser_args,
        stdout=logfile_handle,
        stderr=subprocess.STDOUT,
    )
    log(f"Browser pid {browser_proc.pid}")

    log("All services running. Press Ctrl+C to stop.")
    exited_pids: set[int] = set()
    try:
        while True:
            for proc in processes:
                if proc.poll() is not None and proc.pid not in exited_pids:
                    exited_pids.add(proc.pid)
                    log(f"WARNING: process pid {proc.pid} exited with code {proc.returncode}")
            time.sleep(5)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        shutdown_all()
        logfile_handle.close()


if __name__ == "__main__":
    main()
