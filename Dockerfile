FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

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

# Crow header-only library.
RUN curl -L -o /usr/local/include/crow.h \
    https://raw.githubusercontent.com/CrowCpp/Crow/v1.0/include/crow.h

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

RUN cmake -S cpp -B build/cpp \
    && cmake --build build/cpp -j

EXPOSE 8091
CMD ["./build/cpp/butterflyc_server"]
