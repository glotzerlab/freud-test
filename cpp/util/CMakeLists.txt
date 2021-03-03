add_library(_util OBJECT diagonalize.h diagonalize.cc)

# We treat the extern folder as a SYSTEM library to avoid getting any diagnostic
# information from it. In particular, this avoids clang-tidy throwing errors due
# to any issues in external code.
target_include_directories(_util SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/extern/)
