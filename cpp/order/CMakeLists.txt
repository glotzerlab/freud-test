add_library(
  _order OBJECT
  Cubatic.cc
  Cubatic.h
  HexaticTranslational.cc
  HexaticTranslational.h
  Nematic.cc
  Nematic.h
  RotationalAutocorrelation.cc
  RotationalAutocorrelation.h
  SolidLiquid.cc
  SolidLiquid.h
  Steinhardt.cc
  Steinhardt.h
  Wigner3j.cc
  Wigner3j.h)

# We treat the extern folder as a SYSTEM library to avoid getting any diagnostic
# information from it. In particular, this avoids clang-tidy throwing errors due
# to any issues in external code.
target_include_directories(_order SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/extern/)

target_include_directories(_order PUBLIC ${PROJECT_SOURCE_DIR}/cpp/cluster)
