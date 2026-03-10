#  Butterfly C


> 蝴蝶识别系统
- 训练默认切到 PyTorch
- 5090 推荐直接用 `uv`
- 模型默认采用 EfficientNetB0
- Flask 管理页面
- C++ 基于 crow 和 asio 重构中


> 训练链路现在优先适配 RTX 5090
> 原 TensorFlow SavedModel / C++ 推理链路保留
> 但训练默认已经不再走 TensorFlow



#### Start Butterfly C

- 下载

```bash
git clone https://github.com/Chenpeel/ButterflyC.git
cd ButterflyC
```

- 依赖库

```bash
uv sync
```

- 运行页面

```bash
uv run python app/app.py
```



#### Train on 5090

- 下载数据集

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

- 训练产物

```bash
main/models/ButterflyC-init.pt
main/models/ButterflyC.pt
main/models/checkpoint.pt
main/models/labels.json
main/models/model_manifest.json
```



#### TensorFlow Export

- 如果你已经有 `.keras` 模型

```bash
uv run python -m main.export_serving --model-name ButterflyC
```

- 说明

```bash
PyTorch 训练默认输出 .pt
TensorFlow SavedModel 导出仍然要求已有 .keras
```



#### Docker

- 导出 ONNX

```bash
uv run python -m main.export_onnx --model-name ButterflyC
```

- 启动 C++ 服务

```bash
docker compose up --build
```

- 测试

```bash
curl -s http://127.0.0.1:8091/healthz
curl -s -F "file=@/path/to/test.jpg" http://127.0.0.1:8091/ur
```



#### C++ Service

- 依赖

```bash
需要先安装 Crow + ONNX Runtime + OpenCV
或保证 CMake 能找到 CrowConfig.cmake / crow.h
```

- 构建

```bash
cmake -S cpp -B build/cpp
cmake --build build/cpp
```

- 运行

```bash
./build/cpp/butterflyc_server
```

- ONNX 导出

```bash
uv run python -m main.export_onnx --model-name ButterflyC
```



#### Download Dataset

[123云盘](https://www.123865.com/s/LwbWTd-B8Ii3)

提取码: `chen`
