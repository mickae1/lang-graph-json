// Project domain: datasets and projects (a project must exist before jobs/reports).
#include "ai_metadata.h"
#include "json_response.h"

#include <string>

namespace example {

AI_METADATA({
    "name": "list_datasets",
    "method": "GET",
    "path": "/api/datasets",
    "domain": "project",
    "description": "Lists the datasets known to the application.",
    "parameters": {
        "filter": {"type": "string", "description": "Optional substring filter on the dataset name"}
    },
    "returns": {"datasets": "array of {name, path}"}
})
std::string listDatasets() { return "{\"datasets\":[]}"; }

AI_METADATA({
    "name": "create_project",
    "method": "POST",
    "path": "/api/projects",
    "domain": "project",
    "description": "Creates a new project and opens it. A project is required before any job or report.",
    "parameters": {
        "name": {"type": "string", "required": true}
    },
    "returns": {"projectId": "string"}
})
std::string createProject(const std::string& name) { return toJson({{"projectId", name}}); }

AI_METADATA({
    "name": "import_dataset",
    "method": "POST",
    "path": "/api/projects/{projectId}/datasets",
    "domain": "project",
    "description": "Imports a dataset folder into a project.",
    "parameters": {
        "projectId":   {"type": "string", "required": true},
        "datasetPath": {"type": "string", "required": true}
    },
    "returns": {"datasetId": "string"}
})
std::string importDataset(const std::string& projectId, const std::string& datasetPath) {
    return toJson({{"projectId", projectId}, {"datasetId", datasetPath}});
}

}  // namespace example
