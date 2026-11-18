"""Built-in tools, declared with the same mechanism as user Python tools.

They are always available and can be whitelisted in ``allowed_tools`` like
any other tool.
"""

from __future__ import annotations

from typing import Any

from .metadata import AI_METADATA, ToolContext


@AI_METADATA({
    "name": "wait_for_event",
    "domain": "builtin",
    "read_only": True,
    "description": "Wait until the backend emits an event (e.g. a job completed) instead of polling.",
    "parameters": {
        "event_type": {"type": "string", "required": True, "description": "Backend event type to wait for, e.g. 'job.completed'"},
        "match": {"type": "object", "description": "Fields the event (or its 'data') must contain, e.g. {'jobId': 'J1'}"},
        "timeout_s": {"type": "number", "default": 300, "description": "Maximum wait in seconds (1-3600)"},
    },
})
async def wait_for_event(event_type: str, ctx: ToolContext, match: dict | None = None, timeout_s: float = 300) -> dict:
    timeout_s = min(max(float(timeout_s), 1.0), 3600.0)
    return await ctx.listener.wait_for(event_type, match, timeout_s)


@AI_METADATA({
    "name": "list_shared_files",
    "domain": "builtin",
    "read_only": True,
    "description": "List files available in a shared exchange folder (inbound/outbound files).",
    "parameters": {
        "folder": {"type": "string", "required": True, "description": "Name of a configured shared folder (see POST /config/paths)"},
        "pattern": {"type": "string", "default": "*", "description": "Glob pattern, e.g. '*.csv'"},
    },
})
def list_shared_files(folder: str, ctx: ToolContext, pattern: str = "*") -> dict[str, Any]:
    from pathlib import Path

    root = ctx.shared_folder(folder)
    if not root.is_dir():
        raise ValueError(f"shared folder '{folder}' does not exist")
    if ".." in Path(pattern).parts or Path(pattern).is_absolute():
        raise ValueError("pattern must stay inside the shared folder")
    files = []
    for f in sorted(root.glob(pattern))[:500]:
        if f.is_file() and root in f.resolve().parents:
            st = f.stat()
            files.append({"name": f.relative_to(root).as_posix(), "size": st.st_size, "modified": int(st.st_mtime)})
    return {"folder": folder, "count": len(files), "files": files}


BUILTINS = [wait_for_event, list_shared_files]
