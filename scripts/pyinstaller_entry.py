"""Entry point of the PyInstaller bundle (absolute imports only)."""

import sys

from langgraph_json.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
