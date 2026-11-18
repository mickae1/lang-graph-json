# Python example: tools as Python functions

`my_tools/` is a package of Python tools (`kind: "python"`), declared with the `AI_METADATA` decorator:

| Module | Tool | Shows |
|---|---|---|
| `files.py` | `csv_summary` | synchronous function (runs in a thread), `ctx.shared_file()` sandboxing |
| `jobs.py` | `jobs_overview` | async function calling the backend REST API several times through `ctx.client` |
| `units.py` | `convert_distance` | no `parameters`: the schema is **inferred from the signature** (`Literal` → enum) |

## Use it with the server

1. Make the package importable: point `plugins_dir` at this folder (or copy `my_tools/` into
   `plugins/` of the bundle).
2. Allow the module in `server.json`:

   ```json
   { "python_tool_modules": ["my_tools"], "plugins_dir": "examples/python" }
   ```

3. Add the tools to `tools.json`, by hand or generated:

   ```bash
   python scripts/generate_tools_json.py --source-dir examples/python --lang python --output python_tools.json
   ```

   `callable` (`my_tools.files:csv_summary`, …) is derived from the module path and the decorated function.

4. Whitelist them in an agent (`allowed_tools`).

Plugins are re-imported on every hot-reload (`POST /reload`).

## Use it as a library

Pass the functions directly, without a whitelist or `tools.json`:

```python
from langgraph_json import Runtime
from my_tools.units import convert_distance

async with Runtime(agents={...}, functions=[convert_distance], llm=model) as rt:
    ...
```
