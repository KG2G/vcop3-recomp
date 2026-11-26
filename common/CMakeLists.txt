add_library(vc3_common STATIC xbe.cpp)
target_include_directories(vc3_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(vc3_common PRIVATE -Wall -Wextra)
