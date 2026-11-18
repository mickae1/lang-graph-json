# ai_metadata.pri — run the lang-graph-json parser from a qmake (.pro) project.
#
# Generates tools.json from the AI_METADATA({...}) macros of your sources as a
# pre-build step: the parser runs before compilation, and again whenever a
# source or header changes.
#
# Usage, at the END of your .pro (after SOURCES / HEADERS):
#
#     AI_METADATA_OUTPUT = $$PWD/config/tools.json   # optional (default: $$OUT_PWD/tools.json)
#     include(path/to/lang-graph-json/qmake/ai_metadata.pri)
#
# Optional variables (set them before the include):
#   AI_METADATA_SOURCE_DIRS  directories/files to scan       (default: $$_PRO_FILE_PWD_)
#   AI_METADATA_OUTPUT       tools.json to write             (default: $$OUT_PWD/tools.json)
#   AI_METADATA_PYTHON       Python interpreter, >= 3.10     (default: python on Windows, python3 elsewhere)
#   AI_METADATA_SCRIPT       parser script                   (default: ../scripts/generate_tools_json.py)
#   AI_METADATA_ARGS         extra parser arguments          (default: --lang cpp), e.g. --marker MY_TOOL
#   AI_METADATA_CHECK        committed tools.json to verify; adds "make ai_metadata_check" (CI)
#
# The parser only needs the Python standard library (no pip install).
# Paths are passed as D:/... (forward slashes): Python accepts them from cmd,
# PowerShell, MSYS/Git Bash and Unix shells alike.

isEmpty(AI_METADATA_SOURCE_DIRS): AI_METADATA_SOURCE_DIRS = $$_PRO_FILE_PWD_
isEmpty(AI_METADATA_OUTPUT):      AI_METADATA_OUTPUT = $$OUT_PWD/tools.json
isEmpty(AI_METADATA_SCRIPT):      AI_METADATA_SCRIPT = $$PWD/../scripts/generate_tools_json.py
isEmpty(AI_METADATA_ARGS):        AI_METADATA_ARGS = --lang cpp
isEmpty(AI_METADATA_PYTHON) {
    win32: AI_METADATA_PYTHON = python
    else:  AI_METADATA_PYTHON = python3
}

!exists($$AI_METADATA_SCRIPT): error("ai_metadata.pri: parser not found: $$AI_METADATA_SCRIPT")

AI_METADATA_OUTPUT = $$absolute_path($$AI_METADATA_OUTPUT, $$OUT_PWD)

_ai_sources =
for(dir, AI_METADATA_SOURCE_DIRS) {
    _ai_sources += --source-dir $$shell_quote($$absolute_path($$dir, $$_PRO_FILE_PWD_))
}

# Rebuild tools.json when any compiled source or header changes.
_ai_deps =
for(file, SOURCES): _ai_deps += $$absolute_path($$file, $$_PRO_FILE_PWD_)
for(file, HEADERS): _ai_deps += $$absolute_path($$file, $$_PRO_FILE_PWD_)

_ai_cmd = $$shell_quote($$AI_METADATA_PYTHON) $$shell_quote($$absolute_path($$AI_METADATA_SCRIPT)) $$_ai_sources $$AI_METADATA_ARGS

ai_metadata.target = $$AI_METADATA_OUTPUT
ai_metadata.depends = $$_ai_deps $$AI_METADATA_SCRIPT
ai_metadata.commands = $$_ai_cmd --output $$shell_quote($$AI_METADATA_OUTPUT)
QMAKE_EXTRA_TARGETS += ai_metadata
PRE_TARGETDEPS += $$AI_METADATA_OUTPUT
QMAKE_CLEAN += $$AI_METADATA_OUTPUT

!isEmpty(AI_METADATA_CHECK) {
    ai_metadata_check.commands = $$_ai_cmd --output $$shell_quote($$absolute_path($$AI_METADATA_CHECK, $$_PRO_FILE_PWD_)) --check
    QMAKE_EXTRA_TARGETS += ai_metadata_check
}
