"""Error paths of the generic parser and of its CLI (run in-process)."""

import json

import pytest

from langgraph_json import tools_parser as tp

OK = {"name": "t", "method": "GET", "path": "/a", "description": "d", "parameters": {}}


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"description": "  "}, "description must be a non-empty string"),
        ({"domain": ""}, "domain must be a non-empty string"),
        ({"parameters": []}, "'parameters' must be an object"),
        ({"parameters": {"p": "string"}}, "parameter 'p' must be an object"),
        ({"parameters": {"p": {"enum": []}}}, "enum must be a non-empty list"),
        ({"parameters": {"p": {"required": "yes"}}}, "'required' must be a boolean"),
        ({"kind": "python", "callable": "bad callable", "method": None}, "'callable' must look like"),
        ({"kind": "python", "callable": "m:f", "read_only": "yes"}, "'read_only' must be a boolean"),
    ],
)
def test_validation_messages(patch, message):
    assert any(message in e for e in tp.validate_tool({**OK, **patch}, "f:1"))


def test_payload_must_be_an_object():
    assert tp.validate_tool([1, 2], "f:1") == ["f:1: payload must be a JSON object"]


def test_lexer_edge_cases():
    # escaped quotes inside JSON strings, unterminated single-line string, plain code
    src = (
        'const char* s = "unterminated\n'
        'AI_METADATA({"name": "t", "method": "GET", "path": "/a", "description": "say \\"hi\\"", "parameters": {}})\n'
        "int x = 1;"
    )
    blocks, errors = tp.extract_blocks(src, "a.cpp", tp.PROFILES["cpp"])
    assert errors == [] and json.loads(blocks[0].text)["description"] == 'say "hi"'


def test_python_helpers():
    assert tp._python_literal("[1, 2]") is None
    assert tp._python_literal("{not python") is None
    assert tp.python_module_name(tp.Path("/a/b.py"), tp.Path("/other")) is None
    assert tp.python_module_name(tp.Path("/a/my-mod.py"), tp.Path("/a")) is None


def test_sources_and_profiles(tmp_path):
    res = tp.parse_sources([tmp_path / "missing"])
    assert res.errors == [f"source not found: {tmp_path / 'missing'}"]
    f = tmp_path / "notes.xyz"
    f.write_text("AI_METADATA({})")
    res = tp.parse_sources([f])
    assert res.tools == [] and "no language profile" in res.warnings[0]
    d = tmp_path / "dir.cpp"
    d.mkdir()
    res = tp.parse_sources([tmp_path], lang="cpp")  # a directory named *.cpp is not a file
    assert res.errors == []


def test_unreadable_file(tmp_path, monkeypatch):
    f = tmp_path / "a.cpp"
    f.write_text("x")

    def boom(*a, **k):
        raise OSError("denied")

    monkeypatch.setattr(tp.Path, "read_text", boom)
    res = tp.parse_sources([f], lang="cpp")
    assert "cannot read (denied)" in res.errors[0]


def test_cli_in_process(tmp_path, capsys):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.cpp").write_text('AI_METADATA({"name": "t", "method": "GET", "path": "/a", "description": "d", "parameters": {}})')
    out = tmp_path / "tools.json"
    assert tp.main(["--source-dir", str(src), "--output", str(out), "--ext", "cpp"]) == 0
    assert "warning:" in capsys.readouterr().err  # missing domain
    assert tp.main(["--source-dir", str(src), "--output", str(out), "--check"]) == 0
    assert "up to date" in capsys.readouterr().out
    out.unlink()
    assert tp.main(["--source-dir", str(src), "--output", str(out), "--check"]) == 1
    assert "out of date" in capsys.readouterr().err
    (src / "b.cpp").write_text('AI_METADATA({"name": 1})')
    assert tp.main(["--source-dir", str(src), "--output", str(out)]) == 1
    err = capsys.readouterr().err
    assert "error:" in err and "not written" in err and not out.exists()


def test_whitespace_between_marker_paren_and_brace():
    src = 'AI_METADATA(  \n  {"name": "t", "method": "GET", "path": "/a", "description": "d", "parameters": {}}  \n)'
    blocks, errors = tp.extract_blocks(src, "a.cpp", tp.PROFILES["cpp"])
    assert errors == [] and len(blocks) == 1
