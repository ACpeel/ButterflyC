#  Butterfly C


> 蝴蝶识别系统
- 训练默认走 PyTorch（优先适配 RTX 5090）
- 5090 推荐直接用 `uv`
- 默认模型：EfficientNetB0（模型名：`ButterflyC`）
- Flask 管理页面（`8090`）
- C++ 服务（Crow + ONNX Runtime，`8091`）


> 推荐链路：
> Torch 训练 -> 导出 ONNX -> 启动 C++ 服务



#### Start Butterfly C (Flask)

- 下载

```bash
git clone https://github.com/Chenpeel/ButterflyC.git
cd ButterflyC
```

- 安装依赖

```bash
uv sync
```

- 启动管理页面（默认端口 `8090`）

```bash
uv run python app/app.py
```

- 打开页面

```bash
http://127.0.0.1:8090
```



#### Train on 5090 (Torch)

- 下载数据集（默认落到 `data/`，训练时会自动同步到 `TEMP/data`）

```bash
uv run python -m main.download_dataset
```

- 开始训练

```bash
uv run python -m main.train --model-name ButterflyC
```

- 默认配置

```bash
# config.yml
training_backend: "torch"
device: "auto"
batch_size: 64
precision: "bf16"
```

- 训练产物（默认在 `main/models/`）

```bash
main/models/ButterflyC-init.pt
main/models/ButterflyC.pt
main/models/checkpoint.pt
main/models/labels.json
main/models/model_manifest.json
```



#### Export ONNX (C++ 推理)

- 导出 ONNX

```bash
uv run python -m main.export_onnx --model-name ButterflyC
```

#### Docker (C++ Service)

- 启动 C++ 服务（默认端口 `8091`）

```bash
docker compose up --build
```

- 测试

```bash
curl -s http://127.0.0.1:8091/healthz
curl -s -F "file=@/path/to/test.jpg" http://127.0.0.1:8091/ur
```



#### C++ Service (本地编译)

- 依赖

```bash
Crow
ONNX Runtime
OpenCV
Boost.System + Threads
```

- 构建

```bash
cmake -S cpp -B build/cpp
cmake --build build/cpp
```

- 运行

```bash
PORT=8091 ./build/cpp/butterflyc_server
```



#### Download Dataset

[123云盘](https://www.123865.com/s/LwbWTd-B8Ii3)

提取码: `chen`
