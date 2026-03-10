FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Optional: pass proxy into `docker build` / `docker compose build`.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG ALL_PROXY
ARG all_proxy
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}
ENV ALL_PROXY=${ALL_PROXY}
ENV all_proxy=${all_proxy}

# Use USTC mirror for faster/lower-latency apt in CN networks.
# Keep it on HTTP so we can bootstrap `ca-certificates` inside minimal images.
RUN sed -i \
    -e 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.ustc.edu.cn/ubuntu/|g' \
    -e 's|http://security.ubuntu.com/ubuntu/|http://mirrors.ustc.edu.cn/ubuntu/|g' \
    /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    curl \
    g++ \
    make \
    pkg-config \
    libboost-system-dev \
    libgomp1 \
    libgl1 \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Crow header-only library (needs the full include tree, not just `crow.h`).
ARG CROW_VERSION=v1.0
RUN curl -L -o /tmp/crow.tgz \
        https://github.com/CrowCpp/Crow/archive/refs/tags/${CROW_VERSION}.tar.gz \
    && mkdir -p /tmp/crow-src \
    && tar -C /tmp/crow-src --strip-components=1 -xzf /tmp/crow.tgz \
    && cp -r /tmp/crow-src/include/crow /usr/local/include/ \
    && cp /tmp/crow-src/include/crow.h /usr/local/include/ \
    && rm -rf /tmp/crow.tgz /tmp/crow-src

# ONNX Runtime (CPU). Change ORT_VERSION if needed.
ARG ORT_VERSION=1.18.1
RUN curl -L -o /tmp/onnxruntime.tgz \
    https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -C /opt/onnxruntime --strip-components=1 -xzf /tmp/onnxruntime.tgz \
    && rm /tmp/onnxruntime.tgz

ENV LD_LIBRARY_PATH="/opt/onnxruntime/lib:${LD_LIBRARY_PATH}"

WORKDIR /app
COPY . /app

# Safety: if the build cache ever sneaks into the image, wipe it.
RUN rm -rf /app/build/cpp /app/build/CMakeCache.txt /app/build

RUN cmake -S cpp -B build/cpp \
    && cmake --build build/cpp -j

EXPOSE 8091
CMD ["./build/cpp/butterflyc_server"]
