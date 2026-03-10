import argparse
import csv
import json
import os
import random
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from main.torch_model import (
    build_torch_model,
    count_trainable_parameters,
    freeze_backbone,
    normalize_model_name,
    unfreeze_model,
)
from main.utils.config import load_config
from main.utils.labels import export_label_artifacts
from main.utils.torch_process import load_data, move_batch_to_device
from main.utils.training_monitor import RuntimeSummary, TrainingMonitor


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_training_config(
    *,
    initial_epochs=None,
    fine_tuning_epochs=None,
    batch_size=None,
):
    configs = load_config()
    if initial_epochs is not None:
        configs["initial_epochs"] = initial_epochs
    if fine_tuning_epochs is not None:
        configs["fine_tuning_epochs"] = fine_tuning_epochs
    if batch_size is not None:
        configs["batch_size"] = batch_size
    refresh_per_second = float(configs.get("rich_progress_refresh_per_second", 4) or 4)
    configs["progress_refresh_seconds"] = 1 / max(refresh_per_second, 1.0)
    return configs


def resolve_device(configs):
    device_config = str(configs.get("device", "auto")).lower()
    if device_config == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        capability_score = capability[0] * 10 + capability[1]
        min_capability = int(configs.get("min_cuda_capability", 70) or 70)
        if capability_score < min_capability:
            warnings.warn(
                (
                    f"当前 GPU 计算能力 sm_{capability_score} 低于已安装 PyTorch 支持的最低 "
                    f"sm_{min_capability}，自动切换到 CPU。"
                    " 如需使用该 GPU，请安装匹配架构的 PyTorch 轮子或在 config.yml 中将 device 设为 cpu。"
                )
            )
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_amp_dtype(configs, device):
    precision = str(configs.get("precision", "float32")).lower()
    if device.type != "cuda":
        return None, "float32"

    if precision in {"bf16", "bfloat16"} and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bfloat16"
    if precision in {"fp16", "float16"}:
        return torch.float16, "float16"
    return None, "float32"


def configure_runtime(configs):
    seed = int(configs.get("seed", 42) or 42)
    set_random_seed(seed)

    matmul_precision = str(configs.get("torch_matmul_precision", "high"))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)

    device = resolve_device(configs)
    amp_dtype, precision_label = resolve_amp_dtype(configs, device)
    channels_last = bool(configs.get("torch_channels_last", True))
    compile_enabled = bool(configs.get("torch_compile", False))

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_names = [torch.cuda.get_device_name(device)]
    else:
        device_names = []

    runtime = RuntimeSummary(
        backend="torch",
        device_type=device.type.upper(),
        device_names=device_names,
        precision=precision_label,
        channels_last=channels_last,
        compile_enabled=compile_enabled,
        num_workers=int(configs.get("torch_num_workers", 0) or 0),
        pin_memory=bool(configs.get("torch_pin_memory", True)),
    )
    return device, amp_dtype, runtime


def build_optimizer(model, learning_rate, weight_decay):
    return AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def build_loss(configs):
    return nn.CrossEntropyLoss(
        label_smoothing=float(configs.get("label_smoothing", 0.0) or 0.0)
    )


def build_stage_paths(configs, stage_key):
    csv_log_path = os.path.join(configs["log_dir"], f"{stage_key}.csv")
    checkpoint_path = os.path.join(configs["model_path"], "checkpoint.pt")
    return csv_log_path, checkpoint_path


def append_metrics_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_model_payload(
    model,
    path,
    *,
    model_name,
    class_names,
    image_size,
    metrics=None,
    epoch=None,
):
    model_to_save = getattr(model, "_orig_mod", model)
    payload = {
        "backend": "torch",
        "model_name": model_name,
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "image_size": image_size,
        "state_dict": model_to_save.state_dict(),
        "metrics": metrics or {},
        "epoch": epoch,
    }
    torch.save(payload, path)


def write_training_manifest(configs, *, model_name, final_model_path, checkpoint_path, class_names):
    manifest = {
        "backend": "torch",
        "default_model": model_name,
        "torch_model_path": final_model_path,
        "checkpoint_path": checkpoint_path,
        "labels_path": configs["labels_path"],
        "image_size": configs["image_size"],
        "num_classes": len(class_names),
        "class_names": list(class_names),
    }
    with open(configs["manifest_path"], "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
    return manifest


def maybe_channels_last(model, channels_last):
    if channels_last:
        return model.to(memory_format=torch.channels_last)
    return model


def compile_model_if_needed(model, configs):
    if not bool(configs.get("torch_compile", False)):
        return model
    if not hasattr(torch, "compile"):
        return model
    mode = str(configs.get("torch_compile_mode", "default"))
    return torch.compile(model, mode=mode)


def train_one_epoch(
    model,
    loader,
    *,
    device,
    criterion,
    optimizer,
    scaler,
    amp_dtype,
    channels_last,
    progress,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    amp_enabled = device.type == "cuda" and amp_dtype is not None
    running_loss = 0.0
    running_correct = 0
    running_samples = 0

    for images, labels in loader:
        images, labels = move_batch_to_device(
            images,
            labels,
            device,
            channels_last=channels_last,
        )
        optimizer.zero_grad(set_to_none=True)

        autocast_kwargs = {
            "device_type": device.type,
            "enabled": amp_enabled,
        }
        if amp_dtype is not None:
            autocast_kwargs["dtype"] = amp_dtype

        with torch.autocast(**autocast_kwargs):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct = (predictions == labels).sum().item()

        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_size
        running_loss += loss.item() * batch_size
        running_correct += correct
        running_samples += batch_size

        progress.update_step(
            loss=running_loss / max(1, running_samples),
            accuracy=running_correct / max(1, running_samples),
            learning_rate=optimizer.param_groups[0]["lr"],
        )

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    *,
    device,
    criterion,
    amp_dtype,
    channels_last,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    amp_enabled = device.type == "cuda" and amp_dtype is not None

    for images, labels in loader:
        images, labels = move_batch_to_device(
            images,
            labels,
            device,
            channels_last=channels_last,
        )
        autocast_kwargs = {
            "device_type": device.type,
            "enabled": amp_enabled,
        }
        if amp_dtype is not None:
            autocast_kwargs["dtype"] = amp_dtype

        with torch.autocast(**autocast_kwargs):
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        predictions = outputs.argmax(dim=1)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
    }


def run_training_stage(
    *,
    model,
    model_name,
    dataset_bundle,
    configs,
    monitor,
    device,
    amp_dtype,
    stage_key,
    stage_name,
    epochs,
    learning_rate,
):
    if epochs <= 0:
        monitor.log_skip(stage_name, "epoch 数为 0")
        return {
            "best_val_loss": None,
            "best_val_accuracy": None,
            "last_metrics": {},
        }

    csv_log_path, checkpoint_path = build_stage_paths(configs, stage_key)
    monitor.log_stage_start(
        stage_name=stage_name,
        epochs=epochs,
        learning_rate=learning_rate,
        dataset_bundle=dataset_bundle,
        trainable_params=count_trainable_parameters(model),
        csv_log_path=csv_log_path,
        checkpoint_path=checkpoint_path,
    )

    optimizer = build_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=float(configs.get("weight_decay", 1e-4) or 1e-4),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,
        patience=5,
        min_lr=float(configs.get("min_learning_rate", 5e-5) or 5e-5),
    )
    criterion = build_loss(configs)
    scaler = GradScaler(
        enabled=device.type == "cuda" and amp_dtype == torch.float16
    )
    progress = monitor.build_stage_progress(
        stage_name=stage_name,
        total_epochs=epochs,
        train_steps=dataset_bundle.train_steps,
        refresh_seconds=configs["progress_refresh_seconds"],
    )
    progress.start()

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    last_metrics = {}
    stage_started_at = time.perf_counter()
    channels_last = bool(configs.get("torch_channels_last", True))

    try:
        for epoch in range(1, epochs + 1):
            progress.start_epoch(epoch, optimizer.param_groups[0]["lr"])
            train_metrics = train_one_epoch(
                model,
                dataset_bundle.train_loader,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                amp_dtype=amp_dtype,
                channels_last=channels_last,
                progress=progress,
            )
            val_metrics = evaluate(
                model,
                dataset_bundle.val_loader,
                device=device,
                criterion=criterion,
                amp_dtype=amp_dtype,
                channels_last=channels_last,
            )
            scheduler.step(val_metrics["loss"])

            last_metrics = {
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            epoch_duration = progress.end_epoch(last_metrics)
            monitor.log_epoch(
                stage_name=stage_name,
                epoch=epoch,
                total_epochs=epochs,
                metrics=last_metrics,
                duration=epoch_duration,
            )
            append_metrics_row(
                csv_log_path,
                {
                    "epoch": epoch,
                    **last_metrics,
                },
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_val_accuracy = val_metrics["accuracy"]
                save_model_payload(
                    model,
                    checkpoint_path,
                    model_name=model_name,
                    class_names=dataset_bundle.class_names,
                    image_size=configs["image_size"],
                    metrics=last_metrics,
                    epoch=epoch,
                )
    finally:
        progress.stop()

    monitor.log_stage_end(
        stage_name,
        {
            "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
            "best_val_accuracy": best_val_accuracy,
        },
        elapsed_seconds=time.perf_counter() - stage_started_at,
    )
    return {
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "best_val_accuracy": best_val_accuracy,
        "last_metrics": last_metrics,
    }


def train(
    model_name="ButterflyC",
    *,
    initial_epochs=None,
    fine_tuning_epochs=None,
    batch_size=None,
    sample_limit=None,
    skip_fine_tuning=False,
):
    configs = build_training_config(
        initial_epochs=initial_epochs,
        fine_tuning_epochs=fine_tuning_epochs,
        batch_size=batch_size,
    )
    device, amp_dtype, runtime_summary = configure_runtime(configs)
    resolved_model_name = normalize_model_name(model_name)
    monitor = TrainingMonitor(configs["log_dir"], resolved_model_name)
    if resolved_model_name != model_name:
        monitor.logger.info(
            "未识别的模型名 %s，回退到 %s。",
            model_name,
            resolved_model_name,
        )

    label_encoder = export_label_artifacts(configs["train_csv"], configs["labels_path"])
    dataset_bundle = load_data(
        configs=configs,
        sample_limit=sample_limit,
        label_encoder=label_encoder,
    )

    resolved_model_name, model, head_parameters = build_torch_model(
        resolved_model_name,
        num_classes=len(dataset_bundle.class_names),
        pretrained=True,
    )
    freeze_backbone(model, head_parameters)
    model = model.to(device)
    model = maybe_channels_last(model, runtime_summary.channels_last)
    model = compile_model_if_needed(model, configs)

    monitor.show_training_plan(
        model_name=resolved_model_name,
        configs=configs,
        dataset_bundle=dataset_bundle,
        runtime=runtime_summary,
        sample_limit=sample_limit,
        skip_fine_tuning=skip_fine_tuning,
    )

    initial_result = run_training_stage(
        model=model,
        model_name=resolved_model_name,
        dataset_bundle=dataset_bundle,
        configs=configs,
        monitor=monitor,
        device=device,
        amp_dtype=amp_dtype,
        stage_key="initial_training",
        stage_name="初始训练",
        epochs=int(configs["initial_epochs"]),
        learning_rate=float(configs["learning_rate"]),
    )

    init_model_path = os.path.join(
        configs["model_path"],
        f"{resolved_model_name}-init.pt",
    )
    save_model_payload(
        model,
        init_model_path,
        model_name=resolved_model_name,
        class_names=dataset_bundle.class_names,
        image_size=configs["image_size"],
        metrics=initial_result["last_metrics"],
        epoch=int(configs["initial_epochs"]),
    )
    monitor.logger.info("已保存初始阶段模型: %s", init_model_path)

    fine_result = {"best_val_loss": None, "best_val_accuracy": None, "last_metrics": {}}
    if not skip_fine_tuning and int(configs["fine_tuning_epochs"]) > 0:
        unfreeze_model(model)
        fine_result = run_training_stage(
            model=model,
            model_name=resolved_model_name,
            dataset_bundle=dataset_bundle,
            configs=configs,
            monitor=monitor,
            device=device,
            amp_dtype=amp_dtype,
            stage_key="fine_tuning",
            stage_name="微调训练",
            epochs=int(configs["fine_tuning_epochs"]),
            learning_rate=float(configs["fine_tuning_learning_rate"]),
        )
    else:
        monitor.log_skip("微调训练", "命令行跳过或 fine_tuning_epochs 为 0")

    final_model_path = os.path.join(
        configs["model_path"],
        f"{resolved_model_name}.pt",
    )
    save_model_payload(
        model,
        final_model_path,
        model_name=resolved_model_name,
        class_names=dataset_bundle.class_names,
        image_size=configs["image_size"],
        metrics=fine_result["last_metrics"] or initial_result["last_metrics"],
        epoch=int(configs["initial_epochs"]) + int(configs["fine_tuning_epochs"]),
    )
    manifest = write_training_manifest(
        configs,
        model_name=resolved_model_name,
        final_model_path=final_model_path,
        checkpoint_path=os.path.join(configs["model_path"], "checkpoint.pt"),
        class_names=dataset_bundle.class_names,
    )
    monitor.log_artifacts(
        final_model_path=final_model_path,
        init_model_path=init_model_path,
        checkpoint_path=manifest["checkpoint_path"],
        manifest_path=configs["manifest_path"],
        labels_path=configs["labels_path"],
    )
    monitor.logger.info("训练完成，已写出 PyTorch 模型与清单文件。")
    return initial_result, fine_result


def parse_args():
    parser = argparse.ArgumentParser(description="Train ButterflyC models with PyTorch.")
    parser.add_argument("--model-name", default="ButterflyC")
    parser.add_argument("--initial-epochs", type=int, default=None)
    parser.add_argument("--fine-tuning-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument(
        "--skip-fine-tuning",
        action="store_true",
        help="Only run the frozen-base initial training stage.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model_name,
        initial_epochs=args.initial_epochs,
        fine_tuning_epochs=args.fine_tuning_epochs,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
        skip_fine_tuning=args.skip_fine_tuning,
    )
