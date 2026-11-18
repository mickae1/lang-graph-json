"""Tools working on the shared exchange folders (synchronous code is fine:
it runs in a worker thread)."""

import csv

from langgraph_json.metadata import AI_METADATA, ToolContext


@AI_METADATA({
    "name": "csv_summary",
    "domain": "files",
    "read_only": True,
    "description": "Summarises a CSV file of a shared folder: row count, columns and the first rows.",
    "parameters": {
        "folder": {"type": "string", "required": True, "description": "Shared folder name, e.g. 'inbox'"},
        "file": {"type": "string", "required": True, "description": "File name inside the folder"},
        "sample_rows": {"type": "integer", "default": 5},
    },
    "returns": {"rows": "integer", "columns": "array", "sample": "array"},
})
def csv_summary(folder: str, file: str, ctx: ToolContext, sample_rows: int = 5) -> dict:
    path = ctx.shared_file(folder, file)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        sample, count = [], 0
        for row in reader:
            if count < sample_rows:
                sample.append(row)
            count += 1
    return {"file": file, "rows": count, "columns": reader.fieldnames or [], "sample": sample}
