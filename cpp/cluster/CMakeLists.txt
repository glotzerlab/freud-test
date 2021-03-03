add_library(_cluster OBJECT Cluster.h Cluster.cc ClusterProperties.h
                            ClusterProperties.cc)

# We treat the extern folder as a SYSTEM library to avoid getting any diagnostic
# information from it. In particular, this avoids clang-tidy throwing errors due
# to any issues in external code.
target_include_directories(_cluster SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/extern/)
