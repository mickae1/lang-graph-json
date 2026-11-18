# C++ example: `AI_METADATA` macros

Each REST handler carries the description of its endpoint as a JSON object passed to a no-op macro
(`include/ai_metadata.h`):

```cpp
#define AI_METADATA(...)   // ignored by the compiler

AI_METADATA({
    "name": "run_job",
    "method": "POST",
    "path": "/api/jobs/{jobId}/run",
    "domain": "jobs",
    "description": "Starts a job. Long-running: completion is notified by a 'job.completed' event.",
    "parameters": { "jobId": {"type": "string", "required": true} },
    "returns": {"status": "string"}
})
std::string runJob(const std::string& jobId) { ... }
```

The metadata lives next to the implementation, so it cannot drift: `tools.json` is a build artefact.

## Generate `tools.json`

```bash
python scripts/generate_tools_json.py --source-dir examples/cpp --output config/tools.json
python scripts/generate_tools_json.py --source-dir examples/cpp --output config/tools.json --check   # CI
```

Markers that are commented out (`// ...`, `/* ... */`), inside string literals, or on the `#define` line are
ignored (see `src/WindowPresenter.cpp`).

## Build (the generation runs as a pre-build step)

With **CMake**:

```bash
cmake -S examples/cpp -B build/cpp
cmake --build build/cpp --config Release
./build/cpp/Release/ai_metadata_example      # Linux: ./build/cpp/ai_metadata_example
# build/cpp/tools.json is regenerated whenever a source changes
```

With **qmake** (`ai_metadata_example.pro`, which includes `qmake/ai_metadata.pri`), from Qt Creator or a
shell:

```bash
mkdir build-qmake && cd build-qmake
qmake ../examples/cpp/ai_metadata_example.pro
make                    # MinGW: mingw32-make · MSVC: nmake or jom
make ai_metadata_check  # CI: fails if config/tools.json is out of date
```

In your own Qt project, add these lines at the end of the `.pro`:

```qmake
AI_METADATA_OUTPUT = $$PWD/config/tools.json
include(path/to/lang-graph-json/qmake/ai_metadata.pri)
```

The sources only use the standard library (C++17). A real application would return `QJsonObject`,
`nlohmann::json`, etc.

| File | Domain | Tools |
|---|---|---|
| `src/WindowPresenter.cpp` | `hmi` | open_item, list_windows, close_window, capture_screenshot |
| `src/ProjectPresenter.cpp` | `project` | list_datasets, create_project, import_dataset |
| `src/JobPresenter.cpp` | `jobs` | create_job, run_job, get_job_status |
| `src/ReportPresenter.cpp` | `analysis` | create_report, compute_view, get_view_summary, get_view_sample, export_view, delete_report |
