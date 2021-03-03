# Need to figure out coverage before CYTHON_FLAGS are set.
option(COVERAGE "Enable coverage" OFF)

# Cython flags must be set before we run find_package for Cython since the
# compiler command is created immediately.
set(CYTHON_FLAGS
    "--directive binding=True,boundscheck=False,wraparound=False,embedsignature=True,always_allow_keywords=True"
    CACHE STRING "The directives for Cython compilation.")

if(COVERAGE)
  set(CYTHON_FLAGS
      "${CYTHON_FLAGS},linetrace=True"
      CACHE STRING "The directives for Cython compilation." FORCE)
endif()

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)

include_directories(${NumPy_INCLUDE_DIRS})

# Avoid Cython/Python3.8 minor incompatibility warnings, see
# https://github.com/cython/cython/issues/3474. Note that this option is a bit
# expansive, but it's a temporary fix and we'll be testing on other Python
# versions concurrently so it shouldn't hide any real issues. For backwards
# compatibility with older CMake, I'm using PythonInterp; when we drop support
# for CMake < 3.12, we should switch to find_package(Python).
find_package(PythonInterp REQUIRED)
if(${PYTHON_VERSION_MAJOR} EQUAL 3
   AND ${PYTHON_VERSION_MINOR} EQUAL 8
   AND NOT WIN32)
  add_compile_options("-Wno-deprecated-declarations")
endif()

set(cython_modules_with_cpp
    box
    cluster
    density
    environment
    locality
    order
    parallel
    pmft)

set(cython_modules_without_cpp diffraction interface msd util)

foreach(cython_module ${cython_modules_with_cpp} ${cython_modules_without_cpp})
  add_cython_target(${cython_module} PY3 CXX)
  add_library(${cython_module} MODULE ${${cython_module}})
  # For some reason this command prints the name of the resulting module. Would
  # be good to suppress if possible.
  python_extension_module(${cython_module})

  target_compile_definitions(
    ${cython_module}
    # Avoid deprecation warnings for unsupported NumPy API versions. See
    # https://numpy.org/doc/1.19/reference/c-api/deprecations.html
    PRIVATE "NPY_NO_DEPRECATED_API=NPY_1_10_API_VERSION"
    # Default voro++ verbosity is high.
    PRIVATE "VOROPP_VERBOSE=1")
  if(COVERAGE)
    target_compile_definitions(
      ${cython_module} # Enable line tracing for coverage purposes if requested.
      PRIVATE "CYTHON_TRACE_NOGIL=1")
  endif()

  target_link_libraries(${cython_module} libfreud)

  # Separate logic required for targets with C++ code.
  if("${cython_module}" IN_LIST cython_modules_with_cpp)

    target_include_directories(
      ${cython_module} PRIVATE ${PROJECT_SOURCE_DIR}/cpp/${cython_module})
  endif()

  install(TARGETS ${cython_module} DESTINATION freud)

  # Coverage requires the Cython-compiled C++ files for line coverage.
  if(COVERAGE)
    install(FILES ${${cython_module}} DESTINATION freud)
  endif()

  if(APPLE)
    set_target_properties(${cython_module} PROPERTIES INSTALL_RPATH
                                                      "@loader_path")
  else()
    set_target_properties(${cython_module} PROPERTIES INSTALL_RPATH "\$ORIGIN")
  endif()
  set_target_properties(${cython_module} PROPERTIES INSTALL_RPATH_USE_LINK_PATH
                                                    True)
endforeach()

# The SolidLiquid class has an instance of cluster::Cluster as a member, so
# including the header requires the Cluster.h header. Would prefer to inherit
# this information from the _order library, but that's not possible since we're
# linking to libfreud.
target_include_directories(order PUBLIC ${PROJECT_SOURCE_DIR}/cpp/cluster)
