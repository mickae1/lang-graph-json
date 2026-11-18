// Minimal JSON response helper so the example builds with the standard
// library only (a real application would use QJsonObject, nlohmann/json...).
#pragma once

#include <map>
#include <sstream>
#include <string>

namespace example {

using Fields = std::map<std::string, std::string>;

inline std::string escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

inline std::string toJson(const Fields& fields) {
    std::ostringstream os;
    os << '{';
    bool first = true;
    for (const auto& [k, v] : fields) {
        if (!first) os << ',';
        os << '"' << escape(k) << "\":\"" << escape(v) << '"';
        first = false;
    }
    os << '}';
    return os.str();
}

}  // namespace example
