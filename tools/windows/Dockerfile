# Cross-compiles the Windows build (x86-64, MinGW-w64, POSIX threads) on Linux.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      g++-mingw-w64-x86-64-posix gcc-mingw-w64-x86-64-posix mingw-w64-tools \
      cmake ninja-build meson pkg-config git curl ca-certificates python3 xz-utils file && \
    update-alternatives --set x86_64-w64-mingw32-g++ /usr/bin/x86_64-w64-mingw32-g++-posix && \
    update-alternatives --set x86_64-w64-mingw32-gcc /usr/bin/x86_64-w64-mingw32-gcc-posix && \
    rm -rf /var/lib/apt/lists/*
COPY toolchain.cmake /opt/toolchain.cmake
COPY meson-cross.txt /opt/meson-cross.txt
ENV PREFIX=/opt/win64
# SDL3 (release 3.4.x, built from source like the AppImage)
RUN git clone --depth 1 --branch release-3.4.0 https://github.com/libsdl-org/SDL.git /tmp/sdl || \
    git clone --depth 1 https://github.com/libsdl-org/SDL.git /tmp/sdl
RUN cmake -S /tmp/sdl -B /tmp/sdl/b -G Ninja -DCMAKE_TOOLCHAIN_FILE=/opt/toolchain.cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$PREFIX -DSDL_TESTS=OFF -DSDL_STATIC=ON -DSDL_SHARED=OFF && \
    cmake --build /tmp/sdl/b && cmake --install /tmp/sdl/b && rm -rf /tmp/sdl
# Khronos EGL headers (libepoxy's EGL part needs them; MinGW has none)
RUN git clone --depth 1 https://github.com/KhronosGroup/EGL-Registry.git /tmp/egl && \
    mkdir -p $PREFIX/include && cp -r /tmp/egl/api/EGL /tmp/egl/api/KHR $PREFIX/include/ && rm -rf /tmp/egl
# libepoxy (OpenGL function loader; the code is shared with the Linux build)
RUN CFLAGS="-I$PREFIX/include" git clone --depth 1 --branch 1.5.10 https://github.com/anholt/libepoxy.git /tmp/epoxy && \
    cd /tmp/epoxy && CFLAGS="-I$PREFIX/include" meson setup b --cross-file /opt/meson-cross.txt --prefix=$PREFIX -Dtests=false -Degl=yes -Dglx=no -Dx11=false -Ddefault_library=static && \
    ninja -C b install && rm -rf /tmp/epoxy
# Vulkan headers + loader
RUN git clone --depth 1 --branch v1.4.309 https://github.com/KhronosGroup/Vulkan-Headers.git /tmp/vkh && \
    cmake -S /tmp/vkh -B /tmp/vkh/b -G Ninja -DCMAKE_TOOLCHAIN_FILE=/opt/toolchain.cmake -DCMAKE_INSTALL_PREFIX=$PREFIX && cmake --install /tmp/vkh/b && \
    git clone --depth 1 --branch v1.4.309 https://github.com/KhronosGroup/Vulkan-Loader.git /tmp/vkl && \
    cmake -S /tmp/vkl -B /tmp/vkl/b -G Ninja -DCMAKE_TOOLCHAIN_FILE=/opt/toolchain.cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$PREFIX -DVULKAN_HEADERS_INSTALL_DIR=$PREFIX -DUSE_MASM=OFF -DBUILD_TESTS=OFF && \
    cmake --build /tmp/vkl/b && cmake --install /tmp/vkl/b && rm -rf /tmp/vkh /tmp/vkl
# shaderc (GLSL -> SPIR-V for the Vulkan renderer), static
RUN git clone --depth 1 --branch v2026.4 https://github.com/google/shaderc.git /tmp/shaderc && \
    cd /tmp/shaderc && ./utils/git-sync-deps && \
    cmake -S /tmp/shaderc -B /tmp/shaderc/b -G Ninja -DCMAKE_TOOLCHAIN_FILE=/opt/toolchain.cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$PREFIX -DSHADERC_SKIP_TESTS=ON -DSHADERC_SKIP_EXAMPLES=ON -DSHADERC_SKIP_COPYRIGHT_CHECK=ON \
      -DSPIRV_SKIP_TESTS=ON -DSPIRV_SKIP_EXECUTABLES=ON -DENABLE_GLSLANG_BINARIES=OFF -DSHADERC_ENABLE_SHARED_CRT=OFF && \
    cmake --build /tmp/shaderc/b && cmake --install /tmp/shaderc/b && rm -rf /tmp/shaderc
ENV PKG_CONFIG_LIBDIR=/opt/win64/lib/pkgconfig
# Link the MinGW threads library statically (the .exe must not need libwinpthread-1.dll)
RUN rm -f /usr/x86_64-w64-mingw32/lib/libwinpthread.dll.a /usr/x86_64-w64-mingw32/lib/libpthread.dll.a
