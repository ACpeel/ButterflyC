# C++ Service Skeleton

This directory contains the native web service skeleton that will replace the
current Flask routes. It already compiles with Crow plus Boost.Asio and keeps
the upload and static-file route layout aligned with the Python app.

## Build

```bash
cmake -S cpp -B build/cpp
cmake --build build/cpp
```

## Run

```bash
./build/cpp/butterflyc_server
```

The server defaults to port `8091`. Set `PORT=8090` if you want to bind to the
same port as the Flask service.

## Current status

- `GET /healthz` is ready for health checks.
- `GET /`, `GET /result`, `GET /static/*`, and `GET /uploaded/*` already serve
  the existing frontend files.
- `POST /ur` accepts multipart image uploads and runs native TensorFlow C
  inference when `main/models/model_manifest.json` and the exported SavedModel
  exist.

## Next step

Export serving artifacts before starting the C++ server:

```bash
python -m main.export_serving --model-name ButterflyC
```
