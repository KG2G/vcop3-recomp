# Build environment for the Linux AppImage: Ubuntu 22.04 (glibc 2.35), so the result runs on
# that and anything newer (SteamOS, current Fedora / Arch / Debian 12 ...).
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      g++-12 gcc-12 git ninja-build make pkg-config python3 python3-pip ca-certificates wget file \
      libepoxy-dev libegl-dev libgl-dev libvulkan-dev zlib1g-dev \
      libx11-dev libxext-dev libxrandr-dev libxcursor-dev libxi-dev libxss-dev libxfixes-dev libxtst-dev \
      libxkbcommon-dev libwayland-dev wayland-protocols libdecor-0-dev libdrm-dev libgbm-dev \
      libasound2-dev libpulse-dev libpipewire-0.3-dev libudev-dev libdbus-1-dev libibus-1.0-dev \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
 && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-12 100 \
 && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-12 100
RUN pip3 install --no-cache-dir "cmake>=3.28,<4"
# SDL3 (not packaged in 22.04): bundled into the AppImage as a shared library.
RUN git clone --depth 1 -b release-3.4.20 https://github.com/libsdl-org/SDL.git /src/SDL \
 && cmake -S /src/SDL -B /src/SDL/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSDL_TESTS=OFF -DSDL_EXAMPLES=OFF \
 && cmake --build /src/SDL/build && cmake --install /src/SDL/build && rm -rf /src/SDL
# shaderc (GLSL -> SPIR-V for the Vulkan renderer): linked statically (shaderc_combined).
RUN git clone --depth 1 -b v2026.4 https://github.com/google/shaderc.git /src/shaderc \
 && cd /src/shaderc && python3 utils/git-sync-deps \
 && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSHADERC_SKIP_TESTS=ON -DSHADERC_SKIP_EXAMPLES=ON \
      -DSHADERC_SKIP_COPYRIGHT_CHECK=ON -DSPIRV_SKIP_TESTS=ON -DSPIRV_SKIP_EXECUTABLES=ON -DENABLE_GLSLANG_BINARIES=OFF \
 && cmake --build build && cmake --install build && rm -rf /src/shaderc
RUN ldconfig
# linuxdeploy (makes the AppImage), run without FUSE.
RUN wget -q -O /usr/local/bin/linuxdeploy https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage \
 && chmod +x /usr/local/bin/linuxdeploy
ENV APPIMAGE_EXTRACT_AND_RUN=1
