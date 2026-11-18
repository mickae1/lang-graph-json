cmake_minimum_required(VERSION 3.16)
project(ai_metadata_example CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter)

file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(ai_metadata_example ${EXAMPLE_SOURCES})
target_include_directories(ai_metadata_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Pre-build step: regenerate tools.json from the AI_METADATA macros.
if(Python3_Interpreter_FOUND)
  set(TOOLS_JSON ${CMAKE_CURRENT_BINARY_DIR}/tools.json)
  add_custom_command(
    OUTPUT ${TOOLS_JSON}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/generate_tools_json.py
            --source-dir ${CMAKE_CURRENT_SOURCE_DIR}/src --lang cpp --output ${TOOLS_JSON}
    DEPENDS ${EXAMPLE_SOURCES}
    COMMENT "Generating tools.json from AI_METADATA macros"
    VERBATIM)
  add_custom_target(tools_json ALL DEPENDS ${TOOLS_JSON})
  add_dependencies(ai_metadata_example tools_json)
endif()
