find_package(SDL3 CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(EPOXY REQUIRED IMPORTED_TARGET epoxy)

# Generated code from vc3recomp. Re-run cmake after regenerating.
file(GLOB VC3_GENERATED CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/generated/*.cpp)
if(NOT VC3_GENERATED)
  message(STATUS "vc3: generated/ is empty — linking stub (run vc3recomp first)")
  set(VC3_GENERATED src/generated_stub.cpp)
endif()

add_library(vc3_imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp ${imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer3.cpp)
target_include_directories(vc3_imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(vc3_imgui PUBLIC SDL3::SDL3)

add_library(vc3_runtime STATIC
  src/memory.cpp
  src/settings.cpp
  src/launcher.cpp
  src/install.cpp
  src/dispatch.cpp
  src/loader.cpp
  src/scheduler.cpp
  src/profiler.cpp
  src/hooks.cpp
  src/mmio.cpp
  src/kernel/kernel.cpp
  src/kernel/objects.cpp
  src/kernel/ps.cpp
  src/kernel/mm.cpp
  src/kernel/ke.cpp
  src/kernel/rtl.cpp
  src/kernel/misc.cpp
  src/kernel/io.cpp
  src/gpu/nv2a.cpp
  src/gpu/pgraph.cpp
  src/gpu/gl_backend.cpp
  src/gpu/gl_device.cpp
  src/io/jvs.cpp
  src/io/input.cpp
  src/io/ohci.cpp
  src/io/usb.cpp
  src/io/baseboard.cpp
  src/io/mediaboard.cpp
  src/audio/apu.cpp
  src/audio/aci.cpp
  src/hwwatch.cpp
  src/platform.cpp
  src/privacy.cpp
  src/log.cpp)
target_include_directories(vc3_runtime PUBLIC include PRIVATE src)
target_compile_definitions(vc3_runtime PRIVATE VC3_VERSION="${PROJECT_VERSION}")

# Vulkan renderer (VC3_RENDERER=vulkan): needs the Vulkan headers and shaderc (GLSL -> SPIR-V).
find_package(Vulkan QUIET)
# Release builds link shaderc statically (shaderc_combined, VC3_STATIC_SHADERC=ON).
option(VC3_STATIC_SHADERC "Link shaderc statically (shaderc_combined)" OFF)
if(VC3_STATIC_SHADERC)
  pkg_check_modules(SHADERC IMPORTED_TARGET shaderc_combined)
else()
  pkg_check_modules(SHADERC IMPORTED_TARGET shaderc)
endif()
if(Vulkan_FOUND AND SHADERC_FOUND)
  target_sources(vc3_runtime PRIVATE src/gpu/vk_device.cpp src/gpu/vk_present.cpp)
  # Vulkan structs are written as `VkXxx info{sType}` with the rest zero: not a mistake here.
  set_source_files_properties(src/gpu/vk_device.cpp src/gpu/vk_present.cpp PROPERTIES COMPILE_OPTIONS
                              "-Wno-missing-field-initializers")
  target_compile_definitions(vc3_runtime PRIVATE VC3_HAVE_VULKAN=1)
  target_link_libraries(vc3_runtime PUBLIC Vulkan::Vulkan PkgConfig::SHADERC)
  message(STATUS "vc3: Vulkan renderer enabled")
else()
  target_sources(vc3_runtime PRIVATE src/gpu/vk_device_stub.cpp)
  message(STATUS "vc3: Vulkan headers or shaderc not found - building without the Vulkan renderer")
endif()
target_link_libraries(vc3_runtime PUBLIC vc3_common SDL3::SDL3 Zydis PkgConfig::EPOXY vc3_imgui miniz chdr-static)

# SMAA: SMAA.hlsl embedded as a string (chunks of adjacent raw literals: MSVC limits each
# literal's length), the lookup tables included from the checkout.
file(READ ${smaa_SOURCE_DIR}/SMAA.hlsl VC3_SMAA_SRC)
string(LENGTH "${VC3_SMAA_SRC}" VC3_SMAA_LEN)
set(VC3_SMAA_INC "// Generated from SMAA.hlsl (runtime/CMakeLists.txt).\nstatic const char kSmaaHlsl[] =\n")
set(VC3_SMAA_POS 0)
while(VC3_SMAA_POS LESS VC3_SMAA_LEN)
  string(SUBSTRING "${VC3_SMAA_SRC}" ${VC3_SMAA_POS} 8000 VC3_SMAA_CHUNK)
  string(APPEND VC3_SMAA_INC "R\"SMAA(${VC3_SMAA_CHUNK})SMAA\"\n")
  math(EXPR VC3_SMAA_POS "${VC3_SMAA_POS} + 8000")
endwhile()
string(APPEND VC3_SMAA_INC ";\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/smaa/smaa_hlsl.inc "${VC3_SMAA_INC}")
target_sources(vc3_runtime PRIVATE src/gpu/smaa.cpp)
target_include_directories(vc3_runtime PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/smaa ${smaa_SOURCE_DIR}/Textures)
target_compile_options(vc3_runtime PRIVATE -Wall -Wextra)

# Generated code is huge and machine-written: build it separately, without warnings.
add_library(vc3_generated OBJECT ${VC3_GENERATED})
target_include_directories(vc3_generated PRIVATE include ${PROJECT_SOURCE_DIR}/generated)
target_compile_options(vc3_generated PRIVATE -w)

add_executable(vc3 src/main.cpp $<TARGET_OBJECTS:vc3_generated>)
target_include_directories(vc3 PRIVATE src)
# Kernel exports self-register via static objects; keep them all.
target_link_libraries(vc3 PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,vc3_runtime>)
target_compile_options(vc3 PRIVATE -Wall -Wextra)
if(WIN32)
  # A windowed program (no console; output goes to vc3.log), a 64 MiB stack for every thread
  # (the recompiled code nests deeply; Windows' default is 1 MiB), and the MinGW runtime linked
  # in, so the .exe needs only SDL3.dll and the system's Vulkan loader.
  target_link_options(vc3 PRIVATE -mwindows -Wl,--stack,67108864 -static-libgcc -static-libstdc++
                      -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic)
  target_link_libraries(vc3 PRIVATE winmm)  # timeBeginPeriod
else()
  target_link_options(vc3 PRIVATE -rdynamic)  # lets crash reports name the lifted function (dladdr)
endif()

# The software rasteriser is the hot path: let the compiler vectorise it.
# VC3_NATIVE=OFF (release builds) targets x86-64-v2 instead of this machine's CPU.
option(VC3_NATIVE "Optimise the software rasteriser for this machine's CPU" ON)
if(VC3_NATIVE)
  set_source_files_properties(src/gpu/pgraph.cpp PROPERTIES COMPILE_OPTIONS "-O3;-march=native")
else()
  set_source_files_properties(src/gpu/pgraph.cpp PROPERTIES COMPILE_OPTIONS "-O3;-march=x86-64-v2")
endif()
