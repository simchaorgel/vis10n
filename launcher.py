# vis10n desktop window: opens at once on a splash (the home background and a spinner), starts the
# server alongside, and swaps the window over to the app as soon as the server answers.
# Kept free of the server's heavy imports (torch, models, datasets) so the window appears immediately.
# Run with: python main.py (which hands straight over to run() here)
import ctypes
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import webview

ROOT = Path(__file__).parent
URL = "http://127.0.0.1:8000"

# The native titlebar, recoloured to match the app
TITLEBAR_COLOR = (0x3c, 0x4f, 0x6d)   # --win-bg
TITLEBAR_TEXT = (240, 246, 250)       # --home-text

# The same home background as the app (painted by the same scripts), so the swap to the app is seamless
SPLASH = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; height: 100%; background: #486e88; overflow: hidden; }
  #bg { position: fixed; inset: 0; display: block; width: 100%; height: 100%; }
  .spinner {
    position: fixed; top: 50%; left: 50%;
    width: 44px; height: 44px; margin: -22px 0 0 -22px;
    box-sizing: border-box;
    border: 5px solid rgba(240, 246, 250, 0.25);
    border-top-color: rgba(240, 246, 250, 0.92);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #status {
    position: fixed; top: calc(50% + 44px); left: 0; right: 0;
    text-align: center; color: rgba(240, 246, 250, 0.92);
    font: 600 15px system-ui, sans-serif;
  }
</style>
</head>
<body>
<canvas id="bg"></canvas>
<div class="spinner"></div>
<div id="status"></div>
<script>/*SCRIPTS*/</script>
<script>mountCanvasBg(document.getElementById('bg'), c => paintGrainBg(c, PRESETS.blue));</script>
</body>
</html>
"""


def splash_html():
    scripts = "\n".join((ROOT / "assets" / name).read_text(encoding="utf-8") for name in ("canvas-bg.js", "grain-bg.js"))
    return SPLASH.replace("/*SCRIPTS*/", scripts)


def wait_for_server(url: str, server: subprocess.Popen) -> bool:
    """Blocks until the server answers; False if it exits or doesn't answer within a minute."""
    for _ in range(120):
        if server.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"{url}/api/ping", timeout=1)
            return True
        except OSError:
            time.sleep(0.5)
    return False


def color_titlebar(window):
    """Windows 11: paint the native titlebar and border in the app's colours (ignored on older Windows)."""
    hwnd = ctypes.c_void_p(window.native.Handle.ToInt64())

    def set_attr(attr, value):
        value = ctypes.c_int(value)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(value), ctypes.sizeof(value))

    def colorref(rgb):
        r, g, b = rgb
        return r | (g << 8) | (b << 16)

    set_attr(20, 1)                         # DWMWA_USE_IMMERSIVE_DARK_MODE: light caption buttons
    set_attr(34, colorref(TITLEBAR_COLOR))  # DWMWA_BORDER_COLOR
    set_attr(35, colorref(TITLEBAR_COLOR))  # DWMWA_CAPTION_COLOR
    set_attr(36, colorref(TITLEBAR_TEXT))   # DWMWA_TEXT_COLOR


def run():
    # The window needs the main thread, so the server runs as a child process (which also keeps reload working)
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
        cwd=ROOT,
    )
    try:
        # Starts hidden and is shown once the splash has drawn, so it never appears empty. The background
        # colour is the splash's, for anything that shows before the page does.
        # Below this the focused flow window plus the neighbours peeking in at the edges stop fitting
        window = webview.create_window(
            "vis10n", html=splash_html(), maximized=True, hidden=True,
            min_size=(1400, 800), background_color="#486e88",
        )
        # Also fires while still hidden, so the titlebar is coloured before it's ever seen
        window.events.shown += lambda: color_titlebar(window)

        splash_ready = threading.Event()

        def on_loaded():
            if not splash_ready.is_set():   # loaded fires again when the app replaces the splash
                splash_ready.set()
                window.show()

        window.events.loaded += on_loaded

        def load_app():
            # Show the window anyway if the splash never reports in
            if not splash_ready.wait(2):
                splash_ready.set()
                window.show()
            if wait_for_server(URL, server):
                window.load_url(URL)
            else:
                window.evaluate_js(
                    "document.querySelector('.spinner').remove();"
                    "document.getElementById('status').textContent = 'the server did not start - see the terminal';"
                )

        # Own app id, so the taskbar shows our icon instead of grouping the window under python.exe
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("vis10n")
        # load_app runs on its own thread once the window is up
        webview.start(load_app, icon=str(ROOT / "assets" / "icon.ico"))
    finally:
        # /T also kills the reload worker, which would otherwise keep holding the port
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(server.pid)], capture_output=True)
