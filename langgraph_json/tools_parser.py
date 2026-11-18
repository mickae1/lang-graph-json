"""Generic build-time generator of ``tools.json`` from annotated sources.

The tool metadata lives next to the code that implements each REST endpoint,
as a JSON object passed to a *marker* (``AI_METADATA`` by default)::

    MARKER({ "name": "...", "method": "POST", "path": "/api/...", ... })

The parser is language-agnostic:

* the marker name is configurable (``--marker``);
* a *language profile* (``--lang``) gives the file extensions and the comment
  / string syntax, so markers that are commented out or inside string
  literals are ignored (``cpp``, ``c``, ``csharp``, ``java``, ``js``, ``ts``,
  ``go``, ``rust``, ``kotlin``, ``swift``, ``python``, ``shell``, ``sql``, ``lua``,
  or ``auto`` = choose per file extension);
* ``--include-comments`` also accepts markers written *inside* comments, for
  languages without macros/annotations (``# AI_METADATA({...})``); the
  comment prefixes are stripped from every line of the JSON payload.

Only the JSON object itself is parsed, so the host language never matters
(with ``--lang python`` a Python dict literal — ``True``/``None`` — is accepted).

Two kinds of tools:

* ``"kind": "rest"`` (default) — ``method`` + ``path`` of a backend endpoint;
* ``"kind": "python"`` — ``callable: "package.module:function"``. On a
  decorated Python function (``@AI_METADATA({...})``) ``callable`` is derived
  from the file's module path (relative to ``--python-root``) and the name
  of the decorated function.

Examples: ``examples/cpp`` (C++ macro) and ``examples/python`` (decorator).

Usage::

    python scripts/generate_tools_json.py --source-dir examples/cpp --output config/tools.json
    python scripts/generate_tools_json.py --source-dir src --lang python --include-comments --output tools.json
    python scripts/generate_tools_json.py --source-dir src --output config/tools.json --check     # CI

This module only uses the standard library.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MARKER = "AI_METADATA"
REQUIRED_FIELDS = ("name", "method", "path", "description", "parameters")  # kind "rest"
REQUIRED_PYTHON_FIELDS = ("name", "description")  # kind "python" (callable may be derived)
KINDS = ("rest", "python")
CALLABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$")
_PY_DEF_AFTER = re.compile(r"\s*\)\s*(?:@[^\n]*\n\s*)*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)")
HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
PARAM_TYPES = {"string", "integer", "number", "boolean", "array", "object"}
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Schema version of the generated tools.json (must match config_loader.SCHEMA_VERSIONS).
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Language profiles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LanguageProfile:
    name: str
    extensions: tuple[str, ...]
    line_comments: tuple[str, ...] = ()
    block_comments: tuple[tuple[str, str], ...] = ()
    # String delimiters, longest first (e.g. Python triple quotes).
    strings: tuple[str, ...] = ('"', "'")
    # Lines starting with one of these (after whitespace) are ignored, e.g. the
    # C/C++ "#define MARKER(...)" line itself.
    ignored_line_prefixes: tuple[str, ...] = ()


_C_LIKE = dict(line_comments=("//",), block_comments=(("/*", "*/"),))

PROFILES: dict[str, LanguageProfile] = {
    p.name: p
    for p in (
        LanguageProfile("cpp", (".cpp", ".cc", ".cxx", ".c++", ".h", ".hh", ".hpp", ".hxx", ".ipp"), ignored_line_prefixes=("#",), **_C_LIKE),
        LanguageProfile("c", (".c", ".h"), ignored_line_prefixes=("#",), **_C_LIKE),
        LanguageProfile("csharp", (".cs",), ignored_line_prefixes=("#",), **_C_LIKE),
        LanguageProfile("java", (".java",), **_C_LIKE),
        LanguageProfile("kotlin", (".kt", ".kts"), strings=('"""', '"', "'"), **_C_LIKE),
        LanguageProfile("js", (".js", ".mjs", ".cjs", ".jsx"), strings=('"', "'", "`"), **_C_LIKE),
        LanguageProfile("ts", (".ts", ".tsx", ".mts", ".cts"), strings=('"', "'", "`"), **_C_LIKE),
        LanguageProfile("go", (".go",), strings=('"', "'", "`"), **_C_LIKE),
        LanguageProfile("rust", (".rs",), strings=('"',), **_C_LIKE),
        LanguageProfile("swift", (".swift",), strings=('"""', '"'), **_C_LIKE),
        LanguageProfile("python", (".py", ".pyi"), line_comments=("#",), strings=('"""', "'''", '"', "'")),
        LanguageProfile("shell", (".sh", ".bash", ".ps1"), line_comments=("#",), strings=('"', "'")),
        LanguageProfile("sql", (".sql",), line_comments=("--",), block_comments=(("/*", "*/"),), strings=("'",)),
        LanguageProfile("lua", (".lua",), line_comments=("--",), block_comments=(("--[[", "]]"),), strings=('"', "'")),
    )
}
AUTO = "auto"


def profile_for(path: Path, lang: str) -> LanguageProfile | None:
    if lang != AUTO:
        return PROFILES[lang]
    ext = path.suffix.lower()
    for p in PROFILES.values():  # dict order: first profile wins (".h" -> cpp)
        if ext in p.extensions:
            return p
    return None


# --------------------------------------------------------------------------- #
# Lexing: comment and string regions
# --------------------------------------------------------------------------- #
COMMENT, STRING = "comment", "string"


def lex_regions(text: str, profile: LanguageProfile) -> list[tuple[int, int, str]]:
    """Return sorted, non-overlapping ``(start, end, kind)`` comment/string regions."""
    regions: list[tuple[int, int, str]] = []
    line_c = sorted(profile.line_comments, key=len, reverse=True)
    block_c = sorted(profile.block_comments, key=lambda b: len(b[0]), reverse=True)
    strings = sorted(profile.strings, key=len, reverse=True)
    i, n = 0, len(text)
    while i < n:
        matched = False
        for opener, closer in block_c:  # block first: "--[[" must win over "--"
            if text.startswith(opener, i):
                end = text.find(closer, i + len(opener))
                end = n if end == -1 else end + len(closer)
                regions.append((i, end, COMMENT))
                i, matched = end, True
                break
        if matched:
            continue
        for tok in line_c:
            if text.startswith(tok, i):
                end = text.find("\n", i)
                end = n if end == -1 else end
                regions.append((i, end, COMMENT))
                i, matched = end, True
                break
        if matched:
            continue
        for q in strings:
            if text.startswith(q, i):
                j = i + len(q)
                while j < n:
                    if text[j] == "\\" and len(q) == 1:
                        j += 2
                        continue
                    if text.startswith(q, j):
                        j += len(q)
                        break
                    if text[j] == "\n" and len(q) == 1 and q != "`":
                        break  # unterminated single-line string: stop at EOL
                    j += 1
                regions.append((i, min(j, n), STRING))
                i, matched = min(j, n), True
                break
        if not matched:
            i += 1
    return regions


def region_at(regions: list[tuple[int, int, str]], pos: int) -> str | None:
    lo, hi = 0, len(regions) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e, kind = regions[mid]
        if pos < s:
            hi = mid - 1
        elif pos >= e:
            lo = mid + 1
        else:
            return kind
    return None


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    file: str
    line: int
    text: str
    end: int = 0  # index of the closing "}" in the source


@dataclass
class ParseResult:
    tools: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files_scanned: int = 0


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _matching_brace(text: str, start: int) -> int | None:
    """Index of the ``}`` closing the JSON object opened at ``start``."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _strip_comment_prefixes(payload: str, profile: LanguageProfile) -> str:
    """Remove ``//``, ``#``, `` * `` … at the start of each line of a payload
    written inside a comment."""
    tokens = sorted(set(profile.line_comments) | {"*"}, key=len, reverse=True)
    out = []
    for k, line in enumerate(payload.split("\n")):
        if k:
            s = line.lstrip()
            for tok in tokens:
                if s.startswith(tok):
                    s = s[len(tok) :]
                    break
            line = s
        out.append(line)
    return "\n".join(out)


def extract_blocks(
    text: str,
    filename: str = "<string>",
    profile: LanguageProfile = PROFILES["cpp"],
    marker: str = DEFAULT_MARKER,
    include_comments: bool = False,
) -> tuple[list[Block], list[str]]:
    """Return the JSON payload of each ``marker({...})`` occurrence."""
    blocks: list[Block] = []
    errors: list[str] = []
    regions = lex_regions(text, profile)
    for m in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(marker)}\s*\(", text):
        kind = region_at(regions, m.start())
        if kind == STRING or (kind == COMMENT and not include_comments):
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        head = text[line_start : m.start()].lstrip()
        if any(head.startswith(p) for p in profile.ignored_line_prefixes):
            continue  # e.g. "#define AI_METADATA(...)"
        i = m.end()
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "{":
            continue  # not a metadata block (declaration, other usage)
        lineno = _line_of(text, m.start())
        end = _matching_brace(text, i)
        if end is None:
            errors.append(f"{filename}:{lineno}: unterminated {marker} block")
            continue
        j = end + 1
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text) or text[j] != ")":
            errors.append(f"{filename}:{lineno}: expected ')' after the JSON object of {marker}")
            continue
        payload = text[i : end + 1]
        if kind == COMMENT:
            payload = _strip_comment_prefixes(payload, profile)
        blocks.append(Block(filename, lineno, payload, end))
    return blocks, errors


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_tool(data: object, where: str) -> list[str]:
    errs: list[str] = []
    if not isinstance(data, dict):
        return [f"{where}: payload must be a JSON object"]
    kind = data.get("kind", "rest")
    if kind not in KINDS:
        return [f"{where}: invalid kind {kind!r} (expected one of {KINDS})"]
    for f in REQUIRED_FIELDS if kind == "rest" else REQUIRED_PYTHON_FIELDS:
        if f not in data:
            errs.append(f"{where}: missing required field '{f}'")
    if kind == "python":
        if "callable" not in data:
            errs.append(f"{where}: python tool without 'callable' (it can only be derived from a decorated Python function)")
        elif not isinstance(data["callable"], str) or not CALLABLE_RE.match(data["callable"]):
            errs.append(f"{where}: 'callable' must look like 'package.module:function'")
        if "read_only" in data and not isinstance(data["read_only"], bool):
            errs.append(f"{where}: 'read_only' must be a boolean")
    name = data.get("name")
    if name is not None and (not isinstance(name, str) or not NAME_RE.match(name)):
        errs.append(f"{where}: invalid tool name {name!r}")
    method = data.get("method")
    if method is not None and (not isinstance(method, str) or method.upper() not in HTTP_METHODS):
        errs.append(f"{where}: invalid HTTP method {method!r}")
    path = data.get("path")
    if path is not None and (not isinstance(path, str) or not path.startswith("/")):
        errs.append(f"{where}: path must be a string starting with '/'")
    desc = data.get("description")
    if desc is not None and (not isinstance(desc, str) or not desc.strip()):
        errs.append(f"{where}: description must be a non-empty string")
    domain = data.get("domain")
    if domain is not None and (not isinstance(domain, str) or not domain.strip()):
        errs.append(f"{where}: domain must be a non-empty string")
    params = data.get("parameters")
    if params is not None:
        if not isinstance(params, dict):
            errs.append(f"{where}: 'parameters' must be an object")
        else:
            for pname, spec in params.items():
                if not isinstance(spec, dict):
                    errs.append(f"{where}: parameter '{pname}' must be an object")
                    continue
                ptype = spec.get("type", "string")
                if ptype not in PARAM_TYPES:
                    errs.append(f"{where}: parameter '{pname}' has unsupported type {ptype!r}")
                if "enum" in spec and (not isinstance(spec["enum"], list) or not spec["enum"]):
                    errs.append(f"{where}: parameter '{pname}' enum must be a non-empty list")
                if "required" in spec and not isinstance(spec["required"], bool):
                    errs.append(f"{where}: parameter '{pname}' 'required' must be a boolean")
            if isinstance(path, str):
                for pp in re.findall(r"\{([A-Za-z0-9_]+)\}", path):
                    if pp not in params:
                        errs.append(f"{where}: path parameter '{{{pp}}}' is not declared in 'parameters'")
    return errs


def normalise_tool(data: dict) -> dict:
    if data.get("kind", "rest") == "python":
        out = {"name": data["name"], "kind": "python", "callable": data["callable"], "domain": data.get("domain", "general"),
               "description": data["description"].strip()}
        for key in ("parameters", "returns", "read_only"):
            if key in data:
                out[key] = data[key]
        return out
    out = {
        "name": data["name"],
        "method": data["method"].upper(),
        "path": data["path"],
        "domain": data.get("domain", "general"),
        "description": data["description"].strip(),
        "parameters": data.get("parameters", {}),
    }
    if "returns" in data:
        out["returns"] = data["returns"]
    return out


def _python_literal(text: str):
    """Python profile: the payload may be a Python dict literal (True/None...)."""
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, dict) else None


def python_module_name(path: Path, root: Path) -> str | None:
    """Dotted module name of ``path`` relative to the import root ``root``."""
    try:
        rel = path.resolve().relative_to(root.resolve()).with_suffix("")
    except ValueError:
        return None
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    if not parts or not all(p.isidentifier() for p in parts):
        return None
    return ".".join(parts)


def derive_callable(text: str, block: Block, module: str | None) -> str | None:
    """``module:function`` of the function decorated by ``block``."""
    m = _PY_DEF_AFTER.match(text, block.end + 1)
    if not m or not module:
        return None
    return f"{module}:{m.group(1)}"


def parse_sources(
    sources: list[Path],
    lang: str = AUTO,
    marker: str = DEFAULT_MARKER,
    extensions: tuple[str, ...] | None = None,
    include_comments: bool = False,
    root: Path | None = None,
    python_root: Path | None = None,
) -> ParseResult:
    res = ParseResult()
    by_name: dict[str, str] = {}
    by_route: dict[tuple[str, str], str] = {}
    if lang != AUTO and lang not in PROFILES:
        res.errors.append(f"unknown language profile '{lang}' (available: {AUTO}, {', '.join(PROFILES)})")
        return res
    if extensions is None:
        extensions = PROFILES[lang].extensions if lang != AUTO else tuple(e for p in PROFILES.values() for e in p.extensions)
    exts = {e.lower() for e in extensions}

    files: dict[Path, Path] = {}  # file -> import root (python modules)
    for src in sources:
        if src.is_file():
            files.setdefault(src, src.parent)
        elif src.is_dir():
            for p in src.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    files.setdefault(p, src)
        else:
            res.errors.append(f"source not found: {src}")

    for path in sorted(files):
        profile = profile_for(path, lang)
        if profile is None:
            res.warnings.append(f"{path}: no language profile for '{path.suffix}' — skipped (use --lang)")
            continue
        res.files_scanned += 1
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            res.errors.append(f"{path}: cannot read ({exc})")
            continue
        try:
            rel = path.resolve().relative_to((root or Path.cwd()).resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        blocks, errs = extract_blocks(text, rel, profile, marker, include_comments)
        res.errors.extend(errs)
        for b in blocks:
            where = f"{b.file}:{b.line}"
            try:
                data = json.loads(b.text)
            except json.JSONDecodeError as exc:
                data = _python_literal(b.text) if profile.name == "python" else None
                if data is None:
                    res.errors.append(f"{where}: malformed JSON in {marker}: {exc.msg} (source line {b.line + exc.lineno - 1})")
                    continue
            if isinstance(data, dict) and profile.name == "python" and "kind" not in data:
                # Same default as the runtime decorator: REST if it names an endpoint.
                data["kind"] = "rest" if ("method" in data and "path" in data) else "python"
            if isinstance(data, dict) and data.get("kind") == "python" and "callable" not in data and profile.name == "python":
                derived = derive_callable(text, b, python_module_name(path, python_root or files[path]))
                if derived:
                    data["callable"] = derived
            errs = validate_tool(data, where)
            if errs:
                res.errors.extend(errs)
                continue
            tool = normalise_tool(data)
            if tool["name"] in by_name:
                res.errors.append(f"{where}: duplicate tool name '{tool['name']}' (first defined at {by_name[tool['name']]})")
                continue
            route = (tool["method"], tool["path"]) if tool.get("kind", "rest") == "rest" else ("PYTHON", tool["callable"])
            if route in by_route:
                res.errors.append(f"{where}: duplicate route {route[0]} {route[1]} (first defined at {by_route[route]})")
                continue
            if "domain" not in data:
                res.warnings.append(f"{where}: no 'domain' — using 'general'")
            by_name[tool["name"]] = where
            by_route[route] = where
            res.tools.append(tool)
    res.tools.sort(key=lambda t: (t["domain"], t["name"]))
    return res


def render(tools: list[dict]) -> str:
    """Deterministic output (no timestamp) so CI can diff it."""
    doc = {"version": SCHEMA_VERSION, "generated_by": "generate_tools_json.py", "tools": tools}
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate tools.json from MARKER({...}) metadata blocks in source files")
    ap.add_argument("--source-dir", "--source", dest="sources", type=Path, action="append", required=True,
                    help="source directory or file (repeatable)")
    ap.add_argument("--output", type=Path, required=True, help="tools.json to write")
    ap.add_argument("--marker", default=DEFAULT_MARKER, help=f"marker name (default {DEFAULT_MARKER})")
    ap.add_argument("--lang", default=AUTO, choices=[AUTO, *PROFILES], help="language profile (default: auto, by extension)")
    ap.add_argument("--ext", action="append", help="restrict to these file extensions (repeatable)")
    ap.add_argument("--include-comments", action="store_true", help="also accept markers written inside comments")
    ap.add_argument("--python-root", type=Path, default=None,
                    help="import root used to derive 'callable' of decorated Python functions (default: each --source-dir)")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if --output is out of date (CI)")
    args = ap.parse_args(argv)

    exts = tuple(e if e.startswith(".") else "." + e for e in args.ext) if args.ext else None
    res = parse_sources(args.sources, args.lang, args.marker, exts, args.include_comments, python_root=args.python_root)
    for w in res.warnings:
        print(f"warning: {w}", file=sys.stderr)
    for e in res.errors:
        print(f"error: {e}", file=sys.stderr)
    if res.errors:
        print(f"{len(res.errors)} error(s) — {args.output} not written", file=sys.stderr)
        return 1

    content = render(res.tools)
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != content:
            print(f"{args.output} is out of date — run generate_tools_json.py", file=sys.stderr)
            return 1
        print(f"{args.output} is up to date ({len(res.tools)} tools)")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8", newline="\n")
    print(f"wrote {args.output}: {len(res.tools)} tools from {res.files_scanned} files")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
