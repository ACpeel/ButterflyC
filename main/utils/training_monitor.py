import logging
import os
import time
from dataclasses import dataclass

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


LOGGER_NAME = "butterflyc.train"


@dataclass(frozen=True)
class TrainingLoggerBundle:
    logger: logging.Logger
    console: Console
    log_path: str


@dataclass(frozen=True)
class RuntimeSummary:
    backend: str
    device_type: str
    device_names: list[str]
    precision: str
    channels_last: bool
    compile_enabled: bool
    num_workers: int
    pin_memory: bool


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


def configure_training_logger(log_dir, model_name=None):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_stem = model_name.lower() if model_name else "train"
    log_path = os.path.join(log_dir, f"{file_stem}-{timestamp}.log")
    console = Console()

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    rich_handler = RichHandler(
        console=console,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(logging.INFO)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)
    return TrainingLoggerBundle(logger=logger, console=console, log_path=log_path)


class TorchStageProgress:
    def __init__(
        self,
        tracker,
        *,
        stage_name,
        total_epochs,
        total_steps,
        refresh_seconds=0.25,
    ):
        self.tracker = tracker
        self.stage_name = stage_name
        self.total_epochs = max(1, int(total_epochs))
        self.total_steps = max(1, int(total_steps))
        self.refresh_seconds = max(0.05, float(refresh_seconds))
        self.progress = None
        self.epoch_task_id = None
        self.batch_task_id = None
        self.current_epoch = 0
        self.epoch_started_at = None

    def start(self):
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
            refresh_per_second=max(1, int(round(1 / self.refresh_seconds))),
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

    def start_epoch(self, epoch, learning_rate):
        self.current_epoch = epoch
        self.epoch_started_at = time.perf_counter()
        if self.progress is None:
            return

        if self.batch_task_id is not None:
            self.progress.remove_task(self.batch_task_id)
        self.batch_task_id = self.progress.add_task(
            f"[magenta]{self.stage_name} | epoch {epoch}/{self.total_epochs}",
            total=self.total_steps,
            loss="--",
            accuracy="--",
            val_loss="--",
            val_accuracy="--",
            lr=format_learning_rate(learning_rate),
        )

    def update_step(self, *, loss, accuracy, learning_rate):
        if self.progress is None or self.batch_task_id is None:
            return

        self.progress.advance(self.batch_task_id, 1)
        self.progress.update(
            self.batch_task_id,
            loss=format_metric(loss),
            accuracy=format_metric(accuracy),
            lr=format_learning_rate(learning_rate),
        )

    def end_epoch(self, metrics):
        elapsed = 0
        if self.epoch_started_at is not None:
            elapsed = time.perf_counter() - self.epoch_started_at

        if self.progress is None:
            return elapsed

        if self.batch_task_id is not None:
            self.progress.remove_task(self.batch_task_id)
            self.batch_task_id = None

        self.progress.advance(self.epoch_task_id, 1)
        self.progress.update(
            self.epoch_task_id,
            description=(
                f"[cyan]{self.stage_name} | "
                f"epoch {self.current_epoch}/{self.total_epochs}"
            ),
            loss=format_metric(metrics.get("train_loss")),
            accuracy=format_metric(metrics.get("train_accuracy")),
            val_loss=format_metric(metrics.get("val_loss")),
            val_accuracy=format_metric(metrics.get("val_accuracy")),
            lr=format_learning_rate(metrics.get("learning_rate")),
        )
        return elapsed

    def stop(self):
        if self.progress is None:
            return
        if self.batch_task_id is not None:
            self.progress.remove_task(self.batch_task_id)
            self.batch_task_id = None
        self.progress.stop()


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
        table = Table(box=box.SIMPLE_HEAVY, show_header=False)
        table.add_column("字段", style="cyan", no_wrap=True)
        table.add_column("内容", style="white")
        table.add_row("模型", model_name)
        table.add_row(
            "训练阶段",
            f"冻结基座 {configs['initial_epochs']} epochs"
            + (
                f" + 微调 {configs['fine_tuning_epochs']} epochs"
                if not skip_fine_tuning and configs["fine_tuning_epochs"] > 0
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
            "分层抽样" if dataset_bundle.stratified_split else "普通随机切分",
        )
        table.add_row(
            "步数",
            f"train_steps={dataset_bundle.train_steps} | "
            f"val_steps={dataset_bundle.val_steps} | "
            f"batch_size={dataset_bundle.batch_size}",
        )
        table.add_row(
            "运行时",
            f"backend={runtime.backend} | "
            f"device={runtime.device_type} | "
            f"precision={runtime.precision}",
        )
        if runtime.device_names:
            table.add_row("设备", ", ".join(runtime.device_names))
        table.add_row(
            "数据加载",
            f"workers={runtime.num_workers} | "
            f"pin_memory={runtime.pin_memory} | "
            f"channels_last={runtime.channels_last}",
        )
        table.add_row("编译", "torch.compile" if runtime.compile_enabled else "关闭")
        table.add_row("输出", f"model={configs['model_path']} | log={configs['log_dir']}")
        table.add_row("日志文件", self.log_path)

        if sample_limit is not None:
            table.add_row("样本限制", str(sample_limit))

        self.console.print(
            Panel.fit(table, title="训练计划", border_style="cyan")
        )
        self.logger.info(
            "训练计划 | model=%s | train=%s | val=%s | batch=%s | "
            "initial_epochs=%s | fine_tuning_epochs=%s | log=%s",
            model_name,
            dataset_bundle.train_samples,
            dataset_bundle.val_samples,
            dataset_bundle.batch_size,
            configs["initial_epochs"],
            configs["fine_tuning_epochs"],
            self.log_path,
        )

    def build_stage_progress(
        self,
        *,
        stage_name,
        total_epochs,
        train_steps,
        refresh_seconds=0.25,
    ):
        return TorchStageProgress(
            self,
            stage_name=stage_name,
            total_epochs=total_epochs,
            total_steps=train_steps,
            refresh_seconds=refresh_seconds,
        )

    def log_stage_start(
        self,
        *,
        stage_name,
        epochs,
        learning_rate,
        dataset_bundle,
        trainable_params,
        csv_log_path,
        checkpoint_path,
    ):
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
        table.add_row("CSV 日志", csv_log_path)
        table.add_row("Checkpoint", checkpoint_path)
        self.console.print(
            Panel.fit(table, title=f"{stage_name} 开始", border_style="green")
        )
        self.logger.info(
            "阶段开始 | %s | epochs=%s | lr=%s | train_steps=%s | "
            "val_steps=%s | trainable=%s",
            stage_name,
            epochs,
            format_learning_rate(learning_rate),
            dataset_bundle.train_steps,
            dataset_bundle.val_steps,
            trainable_params,
        )

    def log_epoch(self, *, stage_name, epoch, total_epochs, metrics, duration):
        self.logger.info(
            "epoch %s/%s | stage=%s | train_loss=%s | train_acc=%s | "
            "val_loss=%s | val_acc=%s | lr=%s | duration=%s",
            epoch,
            total_epochs,
            stage_name,
            format_metric(metrics.get("train_loss")),
            format_metric(metrics.get("train_accuracy")),
            format_metric(metrics.get("val_loss")),
            format_metric(metrics.get("val_accuracy")),
            format_learning_rate(metrics.get("learning_rate")),
            format_duration(duration),
        )

    def log_stage_end(self, stage_name, metrics, elapsed_seconds):
        self.logger.info(
            "阶段结束 | %s | duration=%s | best_val_loss=%s | best_val_acc=%s",
            stage_name,
            format_duration(elapsed_seconds),
            format_metric(metrics.get("best_val_loss")),
            format_metric(metrics.get("best_val_accuracy")),
        )

    def log_skip(self, stage_name, reason):
        self.logger.info("阶段跳过 | %s | %s", stage_name, reason)

    def log_artifacts(
        self,
        *,
        final_model_path,
        init_model_path,
        checkpoint_path,
        manifest_path,
        labels_path,
    ):
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("产物", style="cyan", no_wrap=True)
        table.add_column("路径", style="white")
        table.add_row("初始模型", init_model_path)
        table.add_row("最终模型", final_model_path)
        table.add_row("Checkpoint", checkpoint_path)
        table.add_row("Manifest", manifest_path)
        table.add_row("Labels", labels_path)
        self.console.print(
            Panel.fit(table, title="训练产物", border_style="blue")
        )
        self.logger.info(
            "训练产物 | final=%s | init=%s | checkpoint=%s | manifest=%s | labels=%s",
            final_model_path,
            init_model_path,
            checkpoint_path,
            manifest_path,
            labels_path,
        )
