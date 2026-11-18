"""qmake integration (qmake/ai_metadata.pri), with a real qmake + make.

Skipped unless qmake and a make tool are found: on PATH, or through
LGJ_QMAKE (path to qmake) and LGJ_MAKE (e.g. mingw32-make, make, nmake, jom).
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from conftest import ROOT

QMAKE = os.environ.get("LGJ_QMAKE") or shutil.which("qmake6") or shutil.which("qmake")
MAKE = os.environ.get("LGJ_MAKE") or shutil.which("mingw32-make") or shutil.which("make")

pytestmark = pytest.mark.skipif(not (QMAKE and MAKE), reason="qmake / make not found (set LGJ_QMAKE and LGJ_MAKE)")

PRO = ROOT / "examples" / "cpp" / "ai_metadata_example.pro"


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600)


def test_qmake_generates_tools_json_before_build(tmp_path):
    r = run([QMAKE, str(PRO)], tmp_path)
    assert r.returncode == 0, r.stderr
    r = run([MAKE], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "wrote" in r.stdout
    generated = (tmp_path / "tools.json").read_text(encoding="utf-8")
    assert generated == (ROOT / "config" / "tools.json").read_text(encoding="utf-8")

    # incremental: nothing changed -> the parser does not run again
    assert "wrote" not in run([MAKE], tmp_path).stdout
    # a source changed -> tools.json is regenerated
    src = ROOT / "examples" / "cpp" / "src" / "JobPresenter.cpp"
    later = time.time() + 5
    os.utime(src, (later, later))
    assert "wrote" in run([MAKE], tmp_path).stdout

    r = run([MAKE, "ai_metadata_check"], tmp_path)
    assert r.returncode == 0 and "up to date" in r.stdout


def test_qmake_check_target_fails_when_stale(tmp_path):
    stale = tmp_path / "stale_tools.json"
    stale.write_text("{}", encoding="utf-8")
    pro = tmp_path / "check.pro"
    pro.write_text(
        "TEMPLATE = aux\n"
        f"AI_METADATA_SOURCE_DIRS = {(ROOT / 'examples' / 'cpp' / 'src').as_posix()}\n"
        f"AI_METADATA_CHECK = {stale.as_posix()}\n"
        f"include({(ROOT / 'qmake' / 'ai_metadata.pri').as_posix()})\n",
        encoding="utf-8",
    )
    build = tmp_path / "build"
    build.mkdir()
    assert run([QMAKE, str(pro)], build).returncode == 0
    r = run([MAKE, "ai_metadata_check"], build)
    assert r.returncode != 0 and "out of date" in r.stderr
