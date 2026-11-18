import json
import subprocess
import sys
from pathlib import Path

import pytest

from langgraph_json import tools_parser as tp

ROOT = Path(__file__).resolve().parent.parent
CPP_EXAMPLE = ROOT / "examples" / "cpp"


# --------------------------------------------------------------------------- #
# C++ example
# --------------------------------------------------------------------------- #
def test_cpp_example_generates_all_tools():
    res = tp.parse_sources([CPP_EXAMPLE], lang="cpp")
    assert res.errors == []
    names = [t["name"] for t in res.tools]
    assert len(names) == 16
    # commented-out / string-literal markers are ignored
    assert "legacy_open" not in names and "old_tool" not in names
    # sorted by domain, then name
    keys = [(t["domain"], t["name"]) for t in res.tools]
    assert keys == sorted(keys)
    run = next(t for t in res.tools if t["name"] == "run_job")
    assert run["method"] == "POST" and run["path"] == "/api/jobs/{jobId}/run"


def test_committed_tools_json_is_in_sync_with_cpp_example():
    res = tp.parse_sources([CPP_EXAMPLE], lang="cpp")
    assert (ROOT / "config" / "tools.json").read_text(encoding="utf-8") == tp.render(res.tools)


def test_define_line_is_ignored():
    src = "#define AI_METADATA(...)\n#  define AI_METADATA(x) x\n"
    blocks, errors = tp.extract_blocks(src, "a.h", tp.PROFILES["cpp"])
    assert blocks == [] and errors == []


def test_nested_braces_and_parens_in_json_strings():
    src = 'AI_METADATA({"name": "t", "method": "GET", "path": "/a", "description": "uses } and ) and {", "parameters": {"p": {"type": "string"}}})\nint x;'
    blocks, errors = tp.extract_blocks(src, "a.cpp", tp.PROFILES["cpp"])
    assert errors == []
    assert json.loads(blocks[0].text)["description"] == "uses } and ) and {"


def test_marker_requires_word_boundary():
    src = 'MY_AI_METADATA({"name": "x"})'
    assert tp.extract_blocks(src, "a.cpp")[0] == []


def test_char_literal_with_quote_does_not_break_lexer():
    src = "char q = '\"';\nAI_METADATA({\"name\": \"t\", \"method\": \"GET\", \"path\": \"/a\", \"description\": \"d\", \"parameters\": {}})"
    blocks, _ = tp.extract_blocks(src, "a.cpp", tp.PROFILES["cpp"])
    assert len(blocks) == 1 and blocks[0].line == 2


# --------------------------------------------------------------------------- #
# Generic: other languages, marker, comments
# --------------------------------------------------------------------------- #
PY_DECORATOR = '''
def AI_METADATA(meta):
    return lambda f: f

@AI_METADATA({
    "name": "py_tool", "method": "GET", "path": "/py", "domain": "demo",
    "description": "python endpoint", "parameters": {}
})
def handler():
    """Docstring mentioning AI_METADATA({"name": "nope"}) is ignored."""
'''

PY_COMMENTS = '''
# AI_METADATA({
#   "name": "commented_tool", "method": "POST", "path": "/c",
#   "description": "declared in a comment", "parameters": {"x": {"type": "integer"}}
# })
def handler(): ...
'''


def test_python_decorator(tmp_path):
    (tmp_path / "api.py").write_text(PY_DECORATOR)
    res = tp.parse_sources([tmp_path], lang="python")
    assert res.errors == []
    assert [t["name"] for t in res.tools] == ["py_tool"]


def test_markers_in_comments_need_flag(tmp_path):
    (tmp_path / "api.py").write_text(PY_COMMENTS)
    assert tp.parse_sources([tmp_path], lang="python").tools == []
    res = tp.parse_sources([tmp_path], lang="python", include_comments=True)
    assert res.errors == []
    assert res.tools[0]["name"] == "commented_tool"
    assert res.tools[0]["parameters"] == {"x": {"type": "integer"}}


def test_block_comment_payload_with_star_prefixes():
    src = """/**
 * AI_METADATA({
 *   "name": "doc_tool", "method": "GET", "path": "/d",
 *   "description": "in a doc comment", "parameters": {}
 * })
 */
public void handler() {}"""
    blocks, errors = tp.extract_blocks(src, "A.java", tp.PROFILES["java"], include_comments=True)
    assert errors == []
    assert json.loads(blocks[0].text)["name"] == "doc_tool"


def test_custom_marker_and_sql_profile(tmp_path):
    (tmp_path / "procs.sql").write_text(
        "-- TOOL({\"name\": \"ignored\"})\n"
        "SELECT 'TOOL({ in a string })';\n"
        "/* TOOL({\"name\": \"block\"}) */\n"
        "CALL TOOL({\"name\": \"sql_tool\", \"method\": \"POST\", \"path\": \"/sql\", \"description\": \"d\", \"parameters\": {}});\n"
    )
    res = tp.parse_sources([tmp_path], lang="sql", marker="TOOL")
    assert res.errors == []
    assert [t["name"] for t in res.tools] == ["sql_tool"]


def test_auto_profile_mixes_languages(tmp_path):
    (tmp_path / "a.cpp").write_text('AI_METADATA({"name": "a", "method": "GET", "path": "/a", "domain": "x", "description": "d", "parameters": {}})')
    (tmp_path / "b.ts").write_text('const s = `AI_METADATA({"name": "fake"})`;\nAI_METADATA({"name": "b", "method": "GET", "path": "/b", "domain": "x", "description": "d", "parameters": {}});')
    (tmp_path / "notes.txt").write_text('AI_METADATA({"name": "c"})')
    res = tp.parse_sources([tmp_path])
    assert res.errors == []
    assert [t["name"] for t in res.tools] == ["a", "b"]


def test_profile_lookup():
    assert tp.profile_for(Path("x.hpp"), "auto").name == "cpp"
    assert tp.profile_for(Path("x.unknown"), "auto") is None
    assert tp.profile_for(Path("x.unknown"), "python").name == "python"


def test_unknown_language_is_an_error(tmp_path):
    assert tp.parse_sources([tmp_path], lang="cobol").errors


# --------------------------------------------------------------------------- #
# Validation errors
# --------------------------------------------------------------------------- #
def _parse_text(tmp_path, text, name="a.cpp"):
    (tmp_path / name).write_text(text)
    return tp.parse_sources([tmp_path], lang="cpp")


def test_malformed_json_reports_line(tmp_path):
    res = _parse_text(tmp_path, "\n\nAI_METADATA({\"name\": \"x\",})")
    assert len(res.errors) == 1 and "a.cpp:3" in res.errors[0] and "malformed JSON" in res.errors[0]


def test_missing_fields_and_bad_values(tmp_path):
    res = _parse_text(tmp_path, 'AI_METADATA({"name": "bad name", "method": "FETCH", "path": "x", "parameters": {"p": {"type": "date"}}})')
    joined = "\n".join(res.errors)
    for needle in ("missing required field 'description'", "invalid tool name", "invalid HTTP method", "path must", "unsupported type"):
        assert needle in joined


def test_undeclared_path_parameter(tmp_path):
    res = _parse_text(tmp_path, 'AI_METADATA({"name": "t", "method": "GET", "path": "/a/{id}", "description": "d", "parameters": {}})')
    assert any("{id}" in e for e in res.errors)


def test_duplicates(tmp_path):
    block = 'AI_METADATA({"name": "%s", "method": "GET", "path": "%s", "description": "d", "parameters": {}})\n'
    res = _parse_text(tmp_path, block % ("t", "/a") + block % ("t", "/b") + block % ("u", "/a"))
    assert any("duplicate tool name 't'" in e for e in res.errors)
    assert any("duplicate route GET /a" in e for e in res.errors)


def test_unterminated_and_missing_paren(tmp_path):
    res = _parse_text(tmp_path, 'AI_METADATA({"name": "t"}\nint x;\nAI_METADATA({"name": ')
    assert any("expected ')'" in e for e in res.errors)
    assert any("unterminated" in e for e in res.errors)


def test_missing_domain_is_a_warning(tmp_path):
    res = _parse_text(tmp_path, 'AI_METADATA({"name": "t", "method": "get", "path": "/a", "description": "d", "parameters": {}})')
    assert res.errors == [] and res.warnings
    assert res.tools[0]["domain"] == "general" and res.tools[0]["method"] == "GET"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_script(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_tools_json.py"), *args], capture_output=True, text=True)


def test_cli_write_then_check(tmp_path):
    out = tmp_path / "tools.json"
    r = run_script("--source-dir", str(CPP_EXAMPLE), "--output", str(out))
    assert r.returncode == 0, r.stderr
    assert len(json.loads(out.read_text())["tools"]) == 16
    assert run_script("--source-dir", str(CPP_EXAMPLE), "--output", str(out), "--check").returncode == 0
    out.write_text("{}")
    r = run_script("--source-dir", str(CPP_EXAMPLE), "--output", str(out), "--check")
    assert r.returncode == 1 and "out of date" in r.stderr


def test_cli_fails_on_errors_without_writing(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.cpp").write_text('AI_METADATA({"name": "x"})')
    out = tmp_path / "tools.json"
    r = run_script("--source-dir", str(tmp_path / "src"), "--output", str(out))
    assert r.returncode == 1 and not out.exists()


def test_generated_file_is_loadable_by_server(tmp_path):
    from langgraph_json.utils.config_loader import load_tools

    out = tmp_path / "tools.json"
    assert tp.main(["--source-dir", str(CPP_EXAMPLE), "--output", str(out)]) == 0
    tools = load_tools(out)
    assert {t.name for t in tools} >= {"open_item", "run_job", "get_view_summary"}


@pytest.mark.parametrize("lang", sorted(tp.PROFILES))
def test_every_profile_finds_a_plain_marker(lang):
    src = 'AI_METADATA({"name": "t", "method": "GET", "path": "/a", "description": "d", "parameters": {}})'
    blocks, errors = tp.extract_blocks(src, "f", tp.PROFILES[lang])
    assert errors == [] and len(blocks) == 1
