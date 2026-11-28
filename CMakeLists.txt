cmake_minimum_required(VERSION 3.24)
project(vcop3_recomp VERSION 0.1.8 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VC3_BUILD_RECOMPILER "Build the XBE -> C++ recompiler" ON)
option(VC3_BUILD_RUNTIME    "Build the runtime + game executable" ON)

include(FetchContent)
set(ZYDIS_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_DOXYGEN  OFF CACHE BOOL "" FORCE)
FetchContent_Declare(zydis
  GIT_REPOSITORY https://github.com/zyantific/zydis.git
  GIT_TAG        v4.1.1
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(zydis)

# Dear ImGui (launcher UI), built below with its SDL3 + SDL_Renderer backends.
FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.92.9b
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(imgui)

# SMAA (anti-aliasing of the finished picture): the reference shader and its lookup tables
# (MIT licence, Jorge Jimenez et al.).
FetchContent_Declare(smaa
  GIT_REPOSITORY https://github.com/iryoku/smaa.git
  GIT_TAG        71c806a838bdd7d517df19192a20f0c61b3ca29d)
FetchContent_MakeAvailable(smaa)

# Game file import: zip (miniz) and MAME CHD (libchdr) readers.
set(MINIZ_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MINIZ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(miniz
  GIT_REPOSITORY https://github.com/richgel999/miniz.git
  GIT_TAG        3.1.2
  GIT_SHALLOW    TRUE)
set(INSTALL_STATIC_LIBS OFF CACHE BOOL "" FORCE)
set(WITH_SYSTEM_ZLIB OFF CACHE BOOL "" FORCE)
FetchContent_Declare(libchdr
  GIT_REPOSITORY https://github.com/rtissera/libchdr.git
  GIT_TAG        v0.3.0
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(miniz libchdr)

# Vulkan functions loaded at runtime (volk, MIT): the program doesn't import them from the Vulkan
# loader, so it still starts where Vulkan is missing or older than 1.3 (it uses OpenGL then).
# (SOURCE_SUBDIR: only the sources -- volk.c is compiled into the runtime.)
FetchContent_Declare(volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG        vulkan-sdk-1.4.309.0
  SOURCE_SUBDIR  none)
FetchContent_MakeAvailable(volk)

# Custom crosshair images: stb_image (Sean Barrett, public domain / MIT).
FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        6b5799bae8a1fe3792691ef4b72ae26434c3e799)
FetchContent_MakeAvailable(stb)

add_subdirectory(common)

if(VC3_BUILD_RECOMPILER)
  add_subdirectory(recompiler)
endif()
if(VC3_BUILD_RUNTIME)
  add_subdirectory(runtime)
endif()
