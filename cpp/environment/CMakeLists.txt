add_library(
  _environment OBJECT
  AngularSeparation.h
  AngularSeparation.cc
  BondOrder.h
  BondOrder.cc
  LocalBondProjection.h
  LocalBondProjection.cc
  LocalDescriptors.h
  LocalDescriptors.cc
  MatchEnv.h
  MatchEnv.cc
  Registration.h)

# We treat the extern folder as a SYSTEM library to avoid getting any diagnostic
# information from it. In particular, this avoids clang-tidy throwing errors due
# to any issues in external code.
target_include_directories(_environment SYSTEM
                           PUBLIC ${PROJECT_SOURCE_DIR}/extern/)
