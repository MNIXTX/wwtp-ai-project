"""
WWTP AI System - 启动器
由 start.bat 调用，也可直接运行: python launcher.py
启动 Streamlit 服务到后台，打开浏览器，写入 PID 文件。
"""
import os
import sys

if sys.platform == 'win32':
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, 'reconfigure'):
            try: s.reconfigure(encoding='utf-8')
            except Exception: pass

import time
import subprocess
import webbrowser
import socket
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8501
BASE_DIR = Path(__file__).parent.resolve()


def is_port_open():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((HOST, PORT)) == 0


def main():
    os.chdir(str(BASE_DIR))

    # 1. Venv check
    venv_python = BASE_DIR / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        print("[Error] venv not found. Run install.bat first.")
        input("Press Enter to exit...")
        sys.exit(1)

    # 2. Already running?
    if is_port_open():
        print(f"[Info] Server already running at http://{HOST}:{PORT}")
        webbrowser.open(f"http://{HOST}:{PORT}")
        return

    # 3. Start Streamlit (prefer pythonw.exe for no console window, fallback to python.exe)
    pythonw_path = venv_python.with_name("pythonw.exe")
    if not pythonw_path.exists():
        pythonw_path = venv_python  # 某些精简发行版没有 pythonw.exe
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    # Capture startup errors to a log file (EXE builds have no visible console)
    crash_log = BASE_DIR / "logs" / "streamlit_crash.log"
    crash_log.parent.mkdir(exist_ok=True)

    cmd = [
        str(pythonw_path),
        "-m", "streamlit", "run", "ui/app.py",
        "--server.headless=true",
        f"--server.port={PORT}",
        f"--server.address={HOST}",
        "--browser.gatherUsageStats=false",
        "--server.enableXsrfProtection=false",
        "--server.enableCORS=false",
    ]

    with open(crash_log, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )

    # 4. Write PID
    (BASE_DIR / ".server.pid").write_text(str(proc.pid))

    # 5. Wait for ready
    print("Starting server", end="", flush=True)
    for _ in range(30):
        if is_port_open():
            break
        print(".", end="", flush=True)
        time.sleep(1)
    print()

    # 6. Open browser
    webbrowser.open(f"http://{HOST}:{PORT}")
    print(f"Ready: http://{HOST}:{PORT}")
    print("Run stop.bat to stop the server.")


if __name__ == "__main__":
    main()
