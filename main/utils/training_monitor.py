import logging
import os
import time
from dataclasses import dataclass

import tensorflow as tf

try:
    from rich import box
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    box = None
    Console = None
    RichHandler = None
    Panel = None
    Progress = None
    SpinnerColumn = None
    BarColumn = None
    MofNCompleteColumn = None
    TextColumn = None
    TimeElapsedColumn = None
    Table = None
    RICH_AVAILABLE = False


LOGGER_NAME = "butterflyc.train"


@dataclass(frozen=True)
class TrainingLoggerBundle:
    logger: logging.Logger
    console: object | None
    log_path: str


@dataclass(frozen=True)
class RuntimeSummary:
    device_type: str
    device_names: list[str]
    nice_increment: int
    tf_inter_op_threads: int | None
    tf_intra_op_threads: int | None
    gpu_strategy: str


def format_metric(value, digits=4):
    if value is None:
        return "--"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_learning_rate(value):
    if value is None:
        return "--"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.6f}"


def format_duration(seconds):
    total_seconds = max(0, int(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _create_console_handler(console):
    if RICH_AVAILABLE and console is not None:
        handler = RichHandler(
            console=console,
            show_path=False,
            markup=False,
            rich_tracebacks=True,
        )
    else:
        handler = logging.StreamHandler()

    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def _runtime_get(runtime_state, attr_name, mapping_key, default):
    if runtime_state is None:
        return default

    if hasattr(runtime_state, attr_name):
        value = getattr(runtime_state, attr_name)
        return default if value is None else value

    if isinstance(runtime_state, dict):
        value = runtime_state.get(mapping_key, default)
        return default if value is None else value

    return default


def configure_training_logger(log_dir, model_name=None):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_stem = model_name.lower() if model_name else "train"
    log_path = os.path.join(log_dir, f"{file_stem}-{timestamp}.log")
    console = Console(stderr=True) if RICH_AVAILABLE else None

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    logger.addHandler(_create_console_handler(console))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)
    return TrainingLoggerBundle(logger=logger, console=console, log_path=log_path)


def build_overview_panel(
    *,
    model_name,
    configs,
    dataset_bundle,
    runtime_state,
    sample_limit,
    skip_fine_tuning,
    log_path,
):
    if not RICH_AVAILABLE or Table is None or Panel is None:
        return None

    table = Table(box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("字段", style="cyan", no_wrap=True)
    table.add_column("内容", style="white")
    table.add_row("模型", model_name)
    table.add_row(
        "训练阶段",
        f"冻结基座 {configs['initial_epochs']} epochs"
        + (
            f" + 微调 {configs['fine_tuning_epochs']} epochs"
            if not skip_fine_tuning and configs['fine_tuning_epochs'] > 0
            else " + 跳过微调"
        ),
    )
    table.add_row(
        "数据集",
        f"train={dataset_bundle.train_samples} | "
        f"val={dataset_bundle.val_samples} | "
        f"classes={len(dataset_bundle.class_names)}",
    )
    table.add_row(
        "切分策略",
        "分层抽样" if getattr(dataset_bundle, "stratified_split", False) else "普通随机切分",
    )
    table.add_row(
        "步数",
        f"train_steps={dataset_bundle.train_steps} | "
        f"val_steps={dataset_bundle.val_steps} | "
        f"batch_size={dataset_bundle.batch_size}",
    )
    table.add_row(
        "运行时",
        f"device={_runtime_get(runtime_state, 'device_type', 'device', 'unknown')} | "
        f"gpu={_runtime_get(runtime_state, 'gpu_strategy', 'gpu_policy', 'disabled')} | "
        f"nice=+{_runtime_get(runtime_state, 'nice_increment', 'nice_increment', 0)}",
    )
    table.add_row(
        "线程",
        "intra="
        f"{_runtime_get(runtime_state, 'tf_intra_op_threads', 'tf_intra_op_threads', '-')}, "
        "inter="
        f"{_runtime_get(runtime_state, 'tf_inter_op_threads', 'tf_inter_op_threads', '-')}, "
        f"data_parallel={configs.get('tf_data_parallel_calls', '-')}",
    )
    table.add_row(
        "输出",
        f"model={configs['model_path']} | log={configs['log_dir']}",
    )
    table.add_row("日志文件", log_path)

    if sample_limit is not None:
        table.add_row("样本限制", str(sample_limit))

    return Panel.fit(table, title="训练计划", border_style="cyan")


def log_training_overview(
    tracker,
    *,
    model_name,
    configs,
    dataset_bundle,
    runtime_state,
    sample_limit=None,
    skip_fine_tuning=False,
):
    panel = build_overview_panel(
        model_name=model_name,
        configs=configs,
        dataset_bundle=dataset_bundle,
        runtime_state=runtime_state,
        sample_limit=sample_limit,
        skip_fine_tuning=skip_fine_tuning,
        log_path=tracker.log_path,
    )
    if panel is not None and tracker.console is not None:
        tracker.console.print(panel)

    tracker.logger.info(
        "训练计划 | model=%s | train=%s | val=%s | batch=%s | initial_epochs=%s | "
        "fine_tuning_epochs=%s | validation_freq=%s | log=%s",
        model_name,
        dataset_bundle.train_samples,
        dataset_bundle.val_samples,
        dataset_bundle.batch_size,
        configs["initial_epochs"],
        configs["fine_tuning_epochs"],
        configs["validation_freq"],
        tracker.log_path,
    )


def count_trainable_params(model):
    return sum(
        int(tf.keras.backend.count_params(weight))
        for weight in model.trainable_weights
    )


def count_non_trainable_params(model):
    return sum(
        int(tf.keras.backend.count_params(weight))
        for weight in model.non_trainable_weights
    )


def current_learning_rate(model):
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        return None

    learning_rate = getattr(optimizer, "learning_rate", None)
    if learning_rate is None:
        return None

    try:
        return float(tf.keras.backend.get_value(learning_rate))
    except (TypeError, ValueError):
        return None


def log_stage_start(
    tracker,
    *,
    stage_name,
    epochs,
    learning_rate,
    model,
    csv_log_path=None,
    checkpoint_path=None,
):
    trainable_params = count_trainable_params(model)
    frozen_params = count_non_trainable_params(model)
    total_params = trainable_params + frozen_params

    if RICH_AVAILABLE and tracker.console is not None and Table is not None and Panel is not None:
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("字段", style="cyan", no_wrap=True)
        table.add_column("内容", style="white")
        table.add_row("阶段", stage_name)
        table.add_row("epochs", str(epochs))
        table.add_row("learning_rate", format_learning_rate(learning_rate))
        table.add_row(
            "参数",
            f"trainable={trainable_params:,} | "
            f"frozen={frozen_params:,} | total={total_params:,}",
        )
        if csv_log_path:
            table.add_row("CSV 日志", csv_log_path)
        if checkpoint_path:
            table.add_row("Checkpoint", checkpoint_path)
        tracker.console.print(
            Panel.fit(table, title=f"{stage_name} 开始", border_style="green")
        )

    tracker.logger.info(
        "阶段开始 | %s | epochs=%s | lr=%s | trainable=%s | frozen=%s",
        stage_name,
        epochs,
        format_learning_rate(learning_rate),
        trainable_params,
        frozen_params,
    )


def extract_final_metrics(history, model=None):
    if history is None:
        return {}

    metrics = {}
    for key, values in history.history.items():
        if values:
            metrics[key] = values[-1]

    if model is not None:
        metrics["lr"] = current_learning_rate(model)
    return metrics


def log_stage_complete(tracker, *, stage_name, history, elapsed_seconds, model=None):
    metrics = extract_final_metrics(history, model=model)
    if not metrics:
        tracker.logger.info("阶段结束 | %s | 无训练历史可记录", stage_name)
        return

    summary = " | ".join(
        f"{metric}={format_metric(value)}"
        for metric, value in sorted(metrics.items())
    )
    tracker.logger.info(
        "阶段结束 | %s | duration=%s | %s",
        stage_name,
        format_duration(elapsed_seconds),
        summary,
    )


def log_export_summary(
    tracker,
    *,
    keras_model_path,
    manifest_path,
    labels_path,
    saved_model_path=None,
):
    if RICH_AVAILABLE and tracker.console is not None and Table is not None and Panel is not None:
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("产物", style="cyan", no_wrap=True)
        table.add_column("路径", style="white")
        table.add_row("Keras 模型", keras_model_path)
        if saved_model_path:
            table.add_row("SavedModel", saved_model_path)
        table.add_row("Manifest", manifest_path)
        table.add_row("Labels", labels_path)
        tracker.console.print(
            Panel.fit(table, title="训练产物", border_style="blue")
        )

    tracker.logger.info(
        "训练产物 | keras=%s | saved_model=%s | manifest=%s | labels=%s",
        keras_model_path,
        saved_model_path or "--",
        manifest_path,
        labels_path,
    )


class RichTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        tracker,
        *,
        stage_name,
        total_epochs,
        refresh_seconds=0.25,
        total_steps=None,
    ):
        super().__init__()
        self.tracker = tracker
        self.stage_name = stage_name
        self.total_epochs = max(1, int(total_epochs))
        self.refresh_seconds = max(0.05, float(refresh_seconds))
        self.total_steps = int(total_steps or 0)
        self.progress = None
        self.epoch_task_id = None
        self.batch_task_id = None
        self.current_epoch = 0
        self.epoch_started_at = None

    def on_train_begin(self, logs=None):
        if not RICH_AVAILABLE or self.tracker.console is None or Progress is None:
            self.tracker.logger.info("%s 开始。", self.stage_name)
            return

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss={task.fields[loss]}"),
            TextColumn("acc={task.fields[accuracy]}"),
            TextColumn("val_loss={task.fields[val_loss]}"),
            TextColumn("val_acc={task.fields[val_accuracy]}"),
            TextColumn("lr={task.fields[lr]}"),
            TimeElapsedColumn(),
            console=self.tracker.console,
            transient=False,
            refresh_per_second=max(1, round(1 / self.refresh_seconds)),
        )
        self.progress.start()
        self.epoch_task_id = self.progress.add_task(
            f"[cyan]{self.stage_name}",
            total=self.total_epochs,
            loss="--",
            accuracy="--",
            val_loss="--",
            val_accuracy="--",
            lr="--",
        )

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.epoch_started_at = time.perf_counter()

        if self.progress is None:
            return

        steps = int(self.params.get("steps", 0) or self.total_steps or 0)
        batch_total = max(1, steps)
        if self.batch_task_id is not None:
            self.progress.remove_task(self.batch_task_id)
        self.batch_task_id = self.progress.add_task(
            f"[magenta]{self.stage_name} | epoch {self.current_epoch}/{self.total_epochs}",
            total=batch_total,
            loss="--",
            accuracy="--",
            val_loss="--",
            val_accuracy="--",
            lr=format_learning_rate(self._current_learning_rate()),
        )

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        if self.batch_task_id is None or self.progress is None:
            return

        self.progress.advance(self.batch_task_id, 1)
        self.progress.update(
            self.batch_task_id,
            loss=format_metric(self._metric(logs, "loss")),
            accuracy=format_metric(
                self._metric(logs, "accuracy", "acc", "categorical_accuracy")
            ),
            lr=format_learning_rate(self._current_learning_rate()),
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = 0
        if self.epoch_started_at is not None:
            elapsed = time.perf_counter() - self.epoch_started_at

        if self.progress is not None:
            if self.batch_task_id is not None:
                self.progress.remove_task(self.batch_task_id)
                self.batch_task_id = None

            self.progress.advance(self.epoch_task_id, 1)
            self.progress.update(
                self.epoch_task_id,
                description=f"[cyan]{self.stage_name} | epoch {self.current_epoch}/{self.total_epochs}",
                loss=format_metric(self._metric(logs, "loss")),
                accuracy=format_metric(
                    self._metric(logs, "accuracy", "acc", "categorical_accuracy")
                ),
                val_loss=format_metric(self._metric(logs, "val_loss")),
                val_accuracy=format_metric(
                    self._metric(
                        logs,
                        "val_accuracy",
                        "val_acc",
                        "val_categorical_accuracy",
                    )
                ),
                lr=format_learning_rate(self._current_learning_rate()),
            )

        self.tracker.logger.info(
            "epoch %s/%s | stage=%s | loss=%s | acc=%s | val_loss=%s | "
            "val_acc=%s | lr=%s | duration=%s",
            self.current_epoch,
            self.total_epochs,
            self.stage_name,
            format_metric(self._metric(logs, "loss")),
            format_metric(
                self._metric(logs, "accuracy", "acc", "categorical_accuracy")
            ),
            format_metric(self._metric(logs, "val_loss")),
            format_metric(
                self._metric(
                    logs,
                    "val_accuracy",
                    "val_acc",
                    "val_categorical_accuracy",
                )
            ),
            format_learning_rate(self._current_learning_rate()),
            format_duration(elapsed),
        )

    def on_train_end(self, logs=None):
        if self.progress is None:
            return

        if self.batch_task_id is not None:
            self.progress.remove_task(self.batch_task_id)
            self.batch_task_id = None
        self.progress.stop()
        self.progress = None

    @staticmethod
    def _metric(logs, *keys):
        for key in keys:
            if key in logs:
                return logs[key]
        return None

    def _current_learning_rate(self):
        return current_learning_rate(self.model)


class TrainingMonitor:
    def __init__(self, log_dir, model_name=None):
        bundle = configure_training_logger(log_dir, model_name=model_name)
        self.logger = bundle.logger
        self.console = bundle.console
        self.log_path = bundle.log_path
        self.model_name = model_name

    def show_training_plan(
        self,
        *,
        model_name,
        configs,
        dataset_bundle,
        runtime,
        sample_limit=None,
        skip_fine_tuning=False,
    ):
        log_training_overview(
            self,
            model_name=model_name,
            configs=configs,
            dataset_bundle=dataset_bundle,
            runtime_state=runtime,
            sample_limit=sample_limit,
            skip_fine_tuning=skip_fine_tuning,
        )

    def build_progress_callback(
        self,
        *,
        stage_name,
        total_epochs,
        train_steps=None,
        refresh_seconds=0.25,
    ):
        return RichTrainingCallback(
            self,
            stage_name=stage_name,
            total_epochs=total_epochs,
            refresh_seconds=refresh_seconds,
            total_steps=train_steps,
        )

    def log_skip(self, stage_name, reason):
        self.logger.info("%s 跳过 | %s", stage_name, reason)

    def log_stage_start(
        self,
        *,
        stage_name,
        epochs,
        learning_rate,
        dataset_bundle,
        trainable_params,
    ):
        if RICH_AVAILABLE and self.console is not None and Table is not None and Panel is not None:
            table = Table(box=box.SIMPLE, show_header=False)
            table.add_column("字段", style="cyan", no_wrap=True)
            table.add_column("内容", style="white")
            table.add_row("阶段", stage_name)
            table.add_row("epochs", str(epochs))
            table.add_row("learning_rate", format_learning_rate(learning_rate))
            table.add_row(
                "数据",
                f"train_steps={dataset_bundle.train_steps} | "
                f"val_steps={dataset_bundle.val_steps}",
            )
            table.add_row("可训练参数", f"{trainable_params:,}")
            self.console.print(
                Panel.fit(table, title=f"{stage_name} 开始", border_style="green")
            )

        self.logger.info(
            "阶段开始 | %s | epochs=%s | lr=%s | train_steps=%s | val_steps=%s | trainable=%s",
            stage_name,
            epochs,
            format_learning_rate(learning_rate),
            dataset_bundle.train_steps,
            dataset_bundle.val_steps,
            trainable_params,
        )

    def log_stage_end(self, stage_name, history, model, elapsed_seconds):
        log_stage_complete(
            self,
            stage_name=stage_name,
            history=history,
            elapsed_seconds=elapsed_seconds,
            model=model,
        )

    def log_artifacts(self, *, keras_model_path, init_model_path, manifest, manifest_path):
        log_export_summary(
            self,
            keras_model_path=keras_model_path,
            manifest_path=manifest_path,
            labels_path=manifest.get("labels_path"),
            saved_model_path=manifest.get("saved_model_path"),
        )
        self.logger.info("初始阶段模型 | %s", init_model_path)
