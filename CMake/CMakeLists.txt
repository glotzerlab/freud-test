# Add the current directory to the module search path to enable TBB finding.
# List commands don't propagate up, so we have to manually set the parent scope
# variable after.
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake")
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    PARENT_SCOPE)

include(freud-macros.cmake)
