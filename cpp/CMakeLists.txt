if(WIN32)
  # Use add_compile_definitions when dropping support for CMake < 3.12 Force
  # Windows to define M_PI in <cmath>
  add_compile_options(/D_USE_MATH_DEFINES)
  # Prevent Windows from defining min/max as macros
  add_compile_options(/DNOMINMAX)
endif()

add_subdirectory(cluster)
add_subdirectory(density)
add_subdirectory(environment)
add_subdirectory(locality)
add_subdirectory(order)
add_subdirectory(parallel)
add_subdirectory(pmft)
add_subdirectory(util)

add_library(
  libfreud SHARED
  $<TARGET_OBJECTS:_cluster>
  $<TARGET_OBJECTS:_density>
  $<TARGET_OBJECTS:_environment>
  $<TARGET_OBJECTS:_locality>
  $<TARGET_OBJECTS:_order>
  $<TARGET_OBJECTS:_parallel>
  $<TARGET_OBJECTS:_pmft>
  $<TARGET_OBJECTS:_util>)

# The namespaced TBB::tbb currently fails on Windows when discovered via FindTBB
# rather than the config, and it's unclear why. It seems to be related to issues
# with the namespacing.
if(WIN32 AND NOT TBB_USING_CONFIG)
  target_link_libraries(libfreud PUBLIC ${TBB_LIBRARY})
else()
  target_link_libraries(libfreud PUBLIC TBB::tbb)
endif()

# We treat the extern folder as a SYSTEM library to avoid getting any diagnostic
# information from it. In particular, this avoids clang-tidy throwing errors due
# to any issues in external code.
target_include_directories(libfreud SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/extern/)

# CMake will automatically prepend any library name with "lib" when creating
# shared libraries, so a CMake library named libfreud will result in
# liblibfreud.so object. This rename gives the more expected name of
# libfreud.so. Note that we choose not to name the CMake target "freud" to avoid
# conflicting with the project name.
set_target_properties(libfreud PROPERTIES OUTPUT_NAME freud)

# Copy the C++ library into the built version.
install(TARGETS libfreud DESTINATION freud)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  # Copy the compile commands into the root of the project.
  add_custom_command(
    TARGET libfreud
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/compile_commands.json
            ${PROJECT_SOURCE_DIR})
endif()
