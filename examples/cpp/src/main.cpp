// Builds the example to prove that AI_METADATA is transparent for the compiler.
#include <iostream>
#include <string>

namespace example {
std::string openItem(const std::string&, const std::string&);
std::string runJob(const std::string&);
std::string getViewSummary(const std::string&);
}  // namespace example

int main() {
    std::cout << example::openItem("Document", "doc-42") << '\n'
              << example::runJob("job-1") << '\n'
              << example::getViewSummary("v1") << '\n';
    return 0;
}
