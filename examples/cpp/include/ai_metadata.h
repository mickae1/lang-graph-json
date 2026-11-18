// ai_metadata.h -- tool metadata marker for lang-graph-json.
//
// AI_METADATA({...}) is a no-op for the compiler: it only carries the JSON
// description of the REST endpoint implemented right below it. At build time,
// scripts/generate_tools_json.py scans the sources, validates every block and
// generates tools.json, from which the agent server creates its tools.
//
// Required fields: name, method, path, description, parameters.
// Optional fields: domain (default "general"), returns.
// Parameter fields: type (string|integer|number|boolean|array|object),
//                   required, description, enum, default, items.
#pragma once

#define AI_METADATA(...)
