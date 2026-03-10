FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    curl \
    g++ \
    make \
    pkg-config \
    libboost-system-dev \
    && rm -rf /var/lib/apt/lists/*

# Crow header-only library.
RUN curl -L -o /usr/local/include/crow.h \
    https://raw.githubusercontent.com/CrowCpp/Crow/v1.0/include/crow.h

# TensorFlow C API (CPU). Change TF_VERSION if you need to match your SavedModel build.
ARG TF_VERSION=2.18.0
RUN curl -L -o /tmp/libtensorflow.tar.gz \
    https://storage.googleapis.com/tensorflow/versions/${TF_VERSION}/libtensorflow-cpu-linux-x86_64.tar.gz \
    && tar -C /usr/local -xzf /tmp/libtensorflow.tar.gz \
    && ldconfig /usr/local/lib \
    && rm /tmp/libtensorflow.tar.gz

WORKDIR /app
COPY . /app

RUN cmake -S cpp -B build/cpp \
    && cmake --build build/cpp -j

EXPOSE 8091
CMD ["./build/cpp/butterflyc_server"]
