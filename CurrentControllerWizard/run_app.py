import os
import sys
import socket
import threading
import webbrowser


def pick_free_port(preferred: int = 8501) -> int:
    for port in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_app_home() -> str:
    # Portable: always anchor to the folder where the EXE lives (when frozen).
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def find_app_script(app_home: str) -> str:
    candidates = [
        os.path.join(app_home, "CurrentCtrlWizard.py"),
        os.path.join(getattr(sys, "_MEIPASS", ""), "CurrentCtrlWizard.py"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Cannot find CurrentCtrlWizard.py next to the EXE.\n"
        "Your dist folder must include it (use --add-data \"CurrentCtrlWizard.py;.\")."
    )


def run_streamlit(app_path: str, port: int):
    # Force non-dev mode + stable behavior in EXE
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
        "--server.runOnSave=false",
    ]

    try:
        from streamlit.web import cli as stcli
        stcli.main()
    except Exception:
        from streamlit.web.cli import main as st_main
        st_main()


def main():
    app_home = get_app_home()
    os.chdir(app_home)  # IMPORTANT: consistent behavior regardless of where launched from

    app_path = find_app_script(app_home)

    port = pick_free_port(8501)
    url = f"http://127.0.0.1:{port}"

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    run_streamlit(app_path, port)


if __name__ == "__main__":
    main()
