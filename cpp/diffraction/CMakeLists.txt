add_library(_diffraction OBJECT StaticStructureFactorDebye.h
                                StaticStructureFactorDebye.cc)

target_include_directories(_diffraction
                           PUBLIC ${PROJECT_SOURCE_DIR}/cpp/density)
