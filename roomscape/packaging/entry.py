"""PyInstaller entry point (the frozen program starts here)."""

import sys

from roomscape.app import main

if __name__ == "__main__":
    sys.exit(main())
