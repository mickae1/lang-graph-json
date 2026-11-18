#!/usr/bin/env python3
"""Generate tools.json from AI_METADATA({...}) macros (build-time step).

    python scripts/generate_tools_json.py --source-dir src/ --output config/tools.json

The implementation lives in ``langgraph_json/tools_parser.py`` and has no
third-party dependency, so this script runs with any Python >= 3.10.
"""

import importlib.util
import sys
from pathlib import Path

_PARSER = Path(__file__).resolve().parent.parent / "langgraph_json" / "tools_parser.py"
_spec = importlib.util.spec_from_file_location("tools_parser", _PARSER)
tools_parser = importlib.util.module_from_spec(_spec)
sys.modules["tools_parser"] = tools_parser  # required by dataclasses
_spec.loader.exec_module(tools_parser)

if __name__ == "__main__":
    sys.exit(tools_parser.main())
