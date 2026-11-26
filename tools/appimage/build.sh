#!/bin/sh
# Builds dist/VirtuaCop3-recomp-x86_64.AppImage inside the Ubuntu 22.04 container
# (tools/appimage/Dockerfile). Needs docker and the generated code (generated/).
set -e
root=$(cd "$(dirname "$0")/../.." && pwd)
docker build -t vc3-appimage "$root/tools/appimage"
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$root:/src" -w /src vc3-appimage sh -c '
  set -e
  cmake -S . -B build-appimage -G Ninja -DCMAKE_BUILD_TYPE=Release -DVC3_NATIVE=OFF -DVC3_STATIC_SHADERC=ON \
        -DVulkan_INCLUDE_DIR=/src/work/third_party/Vulkan-Headers/include
  cmake --build build-appimage --target vc3
  rm -rf build-appimage/AppDir
  linuxdeploy --appdir build-appimage/AppDir -e build-appimage/runtime/vc3 \
      -d tools/appimage/vc3-recomp.desktop -i tools/appimage/vc3-recomp.png --output appimage
  mkdir -p dist && mv -f Virtua_Cop_3*-x86_64.AppImage dist/VirtuaCop3-recomp-x86_64.AppImage
'
echo "built $root/dist/VirtuaCop3-recomp-x86_64.AppImage"
