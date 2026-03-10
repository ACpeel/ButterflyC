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

- 本地构建

```bash
cd ButterflyC
docker build -t BC . -f docker/Dockerfile
docker run -p 8090:8090 bc:latest
# Warning 是正常的
```

- 浏览器打开

```bash
http://127.0.0.1:8090
```



#### C++ Service

- 依赖

```bash
需要先安装 Crow
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



#### Download Dataset

[123云盘](https://www.123865.com/s/LwbWTd-B8Ii3)

提取码: `chen`
