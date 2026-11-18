# qmake version of the C++ example (same sources as CMakeLists.txt).
#
#   qmake ai_metadata_example.pro && make        (MinGW: mingw32-make, MSVC: nmake / jom)
#   make ai_metadata_check                       verifies ../../config/tools.json is up to date
#
# tools.json is generated in the build directory before compilation,
# by qmake/ai_metadata.pri.

TEMPLATE = app
TARGET = ai_metadata_example
CONFIG += console c++17
CONFIG -= app_bundle qt

INCLUDEPATH += include

HEADERS += \
    include/ai_metadata.h \
    include/json_response.h

SOURCES += \
    src/JobPresenter.cpp \
    src/ProjectPresenter.cpp \
    src/ReportPresenter.cpp \
    src/WindowPresenter.cpp \
    src/main.cpp

# Must come after SOURCES / HEADERS.
AI_METADATA_SOURCE_DIRS = src
AI_METADATA_CHECK = ../../config/tools.json
include(../../qmake/ai_metadata.pri)
