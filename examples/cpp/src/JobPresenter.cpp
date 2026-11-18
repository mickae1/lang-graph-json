// Jobs domain: long-running computations.
#include "ai_metadata.h"
#include "json_response.h"

#include <string>

namespace example {

AI_METADATA({
    "name": "create_job",
    "method": "POST",
    "path": "/api/jobs",
    "domain": "jobs",
    "description": "Creates a computation job in a project.",
    "parameters": {
        "projectId": {"type": "string", "required": true},
        "type":      {"type": "string", "enum": ["Import", "Compute", "Optimize"], "default": "Compute"}
    },
    "returns": {"jobId": "string"}
})
std::string createJob(const std::string& projectId) { return toJson({{"jobId", "job-" + projectId}}); }

AI_METADATA({
    "name": "run_job",
    "method": "POST",
    "path": "/api/jobs/{jobId}/run",
    "domain": "jobs",
    "description": "Starts a job. Long-running: completion is notified by a 'job.completed' event.",
    "parameters": {
        "jobId": {"type": "string", "required": true}
    },
    "returns": {"status": "string"}
})
std::string runJob(const std::string& jobId) { return toJson({{"status", "started"}, {"jobId", jobId}}); }

AI_METADATA({
    "name": "get_job_status",
    "method": "GET",
    "path": "/api/jobs/{jobId}",
    "domain": "jobs",
    "description": "Returns the progress and state of a job.",
    "parameters": {
        "jobId": {"type": "string", "required": true}
    },
    "returns": {"state": "string", "progress": "number"}
})
std::string getJobStatus(const std::string& jobId) { return toJson({{"state", "done"}, {"jobId", jobId}}); }

}  // namespace example
