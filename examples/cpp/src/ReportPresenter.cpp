// Analysis domain: agents work on compact summaries, never on full datasets.
#include "ai_metadata.h"
#include "json_response.h"

#include <string>

namespace example {

AI_METADATA({
    "name": "create_report",
    "method": "POST",
    "path": "/api/reports",
    "domain": "analysis",
    "description": "Creates a report on datasets and/or job results of the open project.",
    "parameters": {
        "name":    {"type": "string", "required": true},
        "type":    {"type": "string", "required": true, "enum": ["Summary", "Trend", "Comparison"]},
        "sources": {"type": "array", "items": {"type": "string"}, "description": "Dataset or job ids"}
    },
    "returns": {"reportId": "string"}
})
std::string createReport(const std::string& name) { return toJson({{"reportId", name}}); }

AI_METADATA({
    "name": "compute_view",
    "method": "POST",
    "path": "/api/reports/{reportId}/views",
    "domain": "analysis",
    "description": "Computes a view (table) of a report.",
    "parameters": {
        "reportId": {"type": "string", "required": true},
        "view":     {"type": "string", "required": true},
        "groupBy":  {"type": "array", "items": {"type": "string"}}
    },
    "returns": {"viewId": "string"}
})
std::string computeView(const std::string& reportId) { return toJson({{"viewId", reportId + "-v1"}}); }

AI_METADATA({
    "name": "get_view_summary",
    "method": "GET",
    "path": "/api/views/{viewId}/summary",
    "domain": "analysis",
    "description": "Returns a compact summary of a computed view: row count, column statistics (min/max/avg) and 10 sample rows.",
    "parameters": {
        "viewId": {"type": "string", "required": true}
    },
    "returns": {"rowCount": "integer", "columns": "array", "sample": "array"}
})
std::string getViewSummary(const std::string& viewId) { return toJson({{"viewId", viewId}, {"rowCount", "0"}}); }

AI_METADATA({
    "name": "get_view_sample",
    "method": "GET",
    "path": "/api/views/{viewId}/sample",
    "domain": "analysis",
    "description": "Returns filtered / sorted rows of a computed view (never the full dataset).",
    "parameters": {
        "viewId": {"type": "string", "required": true},
        "filter": {"type": "string", "description": "Filter expression, e.g. 'value > 10'"},
        "sortBy": {"type": "string"},
        "limit":  {"type": "integer", "default": 20}
    },
    "returns": {"rows": "array"}
})
std::string getViewSample(const std::string& viewId) { return toJson({{"viewId", viewId}}); }

AI_METADATA({
    "name": "export_view",
    "method": "POST",
    "path": "/api/views/{viewId}/export",
    "domain": "analysis",
    "description": "Exports a computed view to CSV or Excel in the outbound shared folder.",
    "parameters": {
        "viewId": {"type": "string", "required": true},
        "format": {"type": "string", "enum": ["csv", "xlsx"], "default": "csv"}
    },
    "returns": {"file": "string"}
})
std::string exportView(const std::string& viewId) { return toJson({{"file", viewId + ".csv"}}); }

AI_METADATA({
    "name": "delete_report",
    "method": "DELETE",
    "path": "/api/reports/{reportId}",
    "domain": "analysis",
    "description": "Deletes a report and its views.",
    "parameters": {
        "reportId": {"type": "string", "required": true}
    }
})
std::string deleteReport(const std::string& reportId) { return toJson({{"deleted", reportId}}); }

}  // namespace example
