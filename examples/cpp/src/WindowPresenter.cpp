// UI domain ("hmi"): windows, items, screenshots.
#include "ai_metadata.h"
#include "json_response.h"

#include <string>

namespace example {

AI_METADATA({
    "name": "open_item",
    "method": "POST",
    "path": "/api/items/open",
    "domain": "hmi",
    "description": "Opens an item in a viewer window, on the requested tab.",
    "parameters": {
        "itemType": {"type": "string", "required": true, "enum": ["Document", "Project", "Dataset"]},
        "itemId":   {"type": "string", "required": true, "description": "Item identifier"},
        "tab":      {"type": "string", "enum": ["Overview", "Details", "History"]}
    },
    "returns": {"windowId": "string", "status": "string"}
})
std::string openItem(const std::string& itemType, const std::string& itemId) {
    return toJson({{"windowId", itemType + ":" + itemId}, {"status", "opened"}});
}

AI_METADATA({
    "name": "list_windows",
    "method": "GET",
    "path": "/api/windows",
    "domain": "hmi",
    "description": "Lists the windows currently open in the application.",
    "parameters": {},
    "returns": {"windows": "array of {windowId, title}"}
})
std::string listWindows() { return "{\"windows\":[]}"; }

AI_METADATA({
    "name": "close_window",
    "method": "DELETE",
    "path": "/api/windows/{windowId}",
    "domain": "hmi",
    "description": "Closes an open window.",
    "parameters": {
        "windowId": {"type": "string", "required": true}
    }
})
std::string closeWindow(const std::string& windowId) { return toJson({{"closed", windowId}}); }

AI_METADATA({
    "name": "capture_screenshot",
    "method": "POST",
    "path": "/api/windows/{windowId}/screenshot",
    "domain": "hmi",
    "description": "Captures a screenshot of a window and returns the path of the PNG file.",
    "parameters": {
        "windowId": {"type": "string", "required": true},
        "scale":    {"type": "number", "default": 1.0, "description": "Scale factor (0.25 - 2.0)"}
    },
    "returns": {"file": "string"}
})
std::string captureScreenshot(const std::string& windowId) { return toJson({{"file", windowId + ".png"}}); }

// Commented-out markers are ignored by the parser:
// AI_METADATA({"name": "legacy_open", "method": "POST", "path": "/api/legacy", "description": "x", "parameters": {}})

/* So are markers inside block comments:
AI_METADATA({"name": "old_tool", "method": "GET", "path": "/api/old", "description": "x", "parameters": {}})
*/

// ...and inside string literals:
const char* kHelp = "Use AI_METADATA({...}) to describe an endpoint.";

}  // namespace example
