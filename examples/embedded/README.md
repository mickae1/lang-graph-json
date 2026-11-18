# Library mode examples

| File | Shows |
|---|---|
| `inprocess.py` | `Runtime(agents=..., functions=[...], llm=...)` with no HTTP server and no files; confirmation handled in code. Uses Claude if `ANTHROPIC_API_KEY` is set, otherwise a scripted fake model. |
| `mounted_fastapi.py` | The full HTTP/WebSocket API mounted under `/agents` in an existing FastAPI application. |

```bash
pip install -e .                       # or the wheel: pip install dist/lang_graph_json-*.whl
python examples/embedded/inprocess.py
uvicorn examples.embedded.mounted_fastapi:app --port 9000
```
