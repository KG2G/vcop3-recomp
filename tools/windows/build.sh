#!/bin/sh
# Builds dist/VirtuaCop3-recomp-windows-x64.exe with MinGW-w64 inside a container
# (tools/windows/Dockerfile). Needs docker and the generated code (generated/).
set -e
root=$(cd "$(dirname "$0")/../.." && pwd)
docker build -t vc3-windows "$root/tools/windows"
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$root:/src" -w /src vc3-windows sh -c '
  set -e
  cmake -S . -B build-windows -G Ninja -DCMAKE_TOOLCHAIN_FILE=/opt/toolchain.cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/opt/win64 -DVC3_NATIVE=OFF -DVC3_STATIC_SHADERC=ON
  cmake --build build-windows --target vc3
  x86_64-w64-mingw32-strip -o build-windows/vc3-stripped.exe build-windows/runtime/vc3.exe
  mkdir -p dist && cp build-windows/vc3-stripped.exe dist/VirtuaCop3-recomp-windows-x64.exe
'
echo "built $root/dist/VirtuaCop3-recomp-windows-x64.exe"
