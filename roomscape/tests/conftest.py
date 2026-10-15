import os

# Headless Qt for GUI smoke tests (no display needed).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
