# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — standalone "--onedir" bundle (Windows and Linux).
#
#   python scripts/build.py            (recommended: build + layout + smoke test)
#   pyinstaller lang-graph-json.spec   (raw build)
#
# Environment variables:
#   LGJ_BUNDLE_MODULES   comma-separated Python packages to embed in the bundle
#                        (your python tools), e.g. "my_tools"; their parent
#                        directory must be importable (see LGJ_BUNDLE_PATHS)
#   LGJ_BUNDLE_PATHS     extra import paths (os.pathsep separated)
#   LGJ_HIDDEN_IMPORTS   extra modules (comma-separated) needed by plugins
import os

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

PACKAGES = [
    "langgraph_json",
    "langgraph",
    "langchain_core",
    "langchain_anthropic",
    "langchain_openai",
    "langchain_ollama",
    "uvicorn",
    "websockets",
    "tiktoken_ext",
    "aiosqlite",  # persistence "sqlite"
]

# Standard-library modules that are not imported by the server itself but are
# commonly used by Python tool plugins loaded from plugins/ at runtime.
PLUGIN_STDLIB = [
    "csv", "sqlite3", "statistics", "decimal", "fractions", "difflib", "zipfile", "tarfile",
    "gzip", "bz2", "lzma", "shutil", "glob", "fnmatch", "calendar", "zoneinfo", "configparser",
    "tomllib", "xml.etree.ElementTree", "xml.dom.minidom", "html", "html.parser", "pprint",
    "textwrap", "string", "random", "secrets", "hashlib", "hmac", "base64", "struct", "uuid",
    "unicodedata", "locale", "gettext", "urllib.request", "urllib.parse", "email", "mimetypes",
    "tempfile", "subprocess", "concurrent.futures", "multiprocessing", "queue", "heapq", "bisect",
    "array", "copy", "pickle", "shelve", "dbm", "plistlib", "getpass", "platform",
]


def _env_list(name, sep=","):
    return [x.strip() for x in os.environ.get(name, "").split(sep) if x.strip()]


bundle_modules = _env_list("LGJ_BUNDLE_MODULES")
hiddenimports = PLUGIN_STDLIB + _env_list("LGJ_HIDDEN_IMPORTS")
for pkg in PACKAGES + bundle_modules:
    hiddenimports += collect_submodules(pkg, filter=lambda name: ".middleware" not in name)

datas = []
for dist in (
    "langgraph", "langchain-core", "langchain-anthropic", "langchain-openai", "langchain-ollama",
    "anthropic", "openai", "ollama", "pydantic", "fastapi", "uvicorn", "httpx", "tiktoken", "lang-graph-json",
    "langgraph-checkpoint-sqlite", "aiosqlite",
):
    try:
        datas += copy_metadata(dist)
    except Exception:  # not installed / no metadata: skip
        pass

a = Analysis(
    ["scripts/pyinstaller_entry.py"],
    pathex=["."] + _env_list("LGJ_BUNDLE_PATHS", os.pathsep),
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # *.middleware need the optional "langchain" package (not a dependency)
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "PyInstaller",
              "langchain_anthropic.middleware", "langchain_openai.middleware"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="lang-graph-json",
    debug=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="lang-graph-json")
