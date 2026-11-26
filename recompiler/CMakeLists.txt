add_executable(vc3recomp
  src/main.cpp
  src/analysis.cpp
  src/emitter.cpp
  src/lifter.cpp)
target_link_libraries(vc3recomp PRIVATE vc3_common Zydis)
target_compile_options(vc3recomp PRIVATE -Wall -Wextra)
