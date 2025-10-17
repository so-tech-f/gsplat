# syntax=docker/dockerfile:1
ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0
# CUDA architectures, required by Colmap and tiny-cuda-nn. Use >= 8.0 for faster TCNN.
ARG CUDA_ARCHITECTURES="90;89;86;80"

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder
ARG CUDA_ARCHITECTURES
ARG NVIDIA_CUDA_VERSION
ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV QT_XCB_GL_INTEGRATION=xcb_egl
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        apt-utils \
        git \
        wget \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libglm-dev \
        libgtest-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev \
        python3.10-dev \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Build and install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.3/cmake-3.31.3-linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.31.3 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.31.3 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.31.3/bin/* /usr/local/bin

# Build and install COLMAP.
RUN git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    git checkout "3.12.6" && \
    mkdir build && \
    cd build && \
    mkdir -p /build && \
    cmake .. -GNinja "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
        -DCMAKE_INSTALL_PREFIX=/build/colmap && \
    ninja install -j1 && \
    cd ~

# Upgrade pip and install dependencies.
# pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118 && \
RUN pip install --no-cache-dir --upgrade pip 'setuptools>=64' && \
    pip install --no-cache-dir torch==2.1.2+cu118 torchvision==0.16.2+cu118 'numpy<2.0.0' --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir pycolmap==0.6.1 pyceres==2.1 omegaconf==2.3.0


# Install gsplat from local copy (for development).
COPY . /gsplat
RUN cd /gsplat && \
    export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \
    export MAX_JOBS=4 && \
    pip install --no-cache-dir -e . --use-pep517 --no-build-isolation && \
    pip install --no-cache-dir -r examples/requirements.txt


# Fix permissions
RUN chmod -R go=u /usr/local/lib/python3.10 && \
    chmod -R go=u /build

#
# Docker runtime stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as runtime
ARG CUDA_ARCHITECTURES
ARG NVIDIA_CUDA_VERSION
ARG UBUNTU_VERSION


# Minimal dependencies to run COLMAP binary compiled in the builder stage.
# Note: this reduces the size of the final image considerably, since all the
# build dependencies are not needed.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        apt-utils \
        libboost-filesystem1.74.0 \
        libboost-program-options1.74.0 \
        libc6 \
        libceres2 \
        libfreeimage3 \
        libgcc-s1 \
        libgl1 \
        libglew2.2 \
        libgoogle-glog0v5 \
        libglm-dev \
        libqt5core5a \
        libqt5gui5 \
        libqt5widgets5 \
        python3.10 \
        python3.10-dev \
        build-essential \
        python-is-python3 \
        ffmpeg \
        vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy packages from builder stage.
COPY --from=builder /build/colmap/ /usr/local/
COPY --from=builder /usr/local/lib/python3.10/dist-packages/ /usr/local/lib/python3.10/dist-packages/
COPY --from=builder /gsplat /gsplat

# Bash as default entrypoint.
CMD /bin/bash -l
