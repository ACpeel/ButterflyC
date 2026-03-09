import argparse
import json
import os
import shutil
import time

STRICT_CONV_PICKER_FLAG = "--xla_gpu_strict_conv_algorithm_picker=false"
existing_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
if "xla_gpu_strict_conv_algorithm_picker" not in existing_xla_flags:
    os.environ["XLA_FLAGS"] = " ".join(
        flag
        for flag in (existing_xla_flags, STRICT_CONV_PICKER_FLAG)
        if flag
    )

import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TerminateOnNaN,
)

from main.model import ButterflyC, DenseNet121M, ResNet50M, VGG16M
from main.utils.config import load_config
from main.utils.labels import export_label_artifacts
from main.utils.process import load_data
from main.utils.training_monitor import RuntimeSummary, TrainingMonitor

MODEL_BUILDERS = {
    "ButterflyC": ButterflyC,
    "VGG16M": VGG16M,
    "ResNet50M": ResNet50M,
    "DenseNet121M": DenseNet121M,
}


def count_trainable_parameters(model):
    return int(
        sum(
            tf.keras.backend.count_params(weight)
            for weight in model.trainable_weights
        )
    )


def configure_runtime(configs):
    nice_increment = int(configs.get("process_nice_increment", 0) or 0)
    inter_op_threads = configs.get("tf_inter_op_threads")
    intra_op_threads = configs.get("tf_intra_op_threads")
    precision = str(configs.get("precision", "float32")).lower()
    gpu_strategy = "CPU-only"

    if precision in {"float16", "mixed_float16"}:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")

    if nice_increment and hasattr(os, "nice"):
        try:
            os.nice(nice_increment)
        except OSError:
            pass

    tf.config.optimizer.set_jit(False)

    if inter_op_threads:
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
    if intra_op_threads:
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return RuntimeSummary(
            device_type="CPU",
            device_names=[],
            nice_increment=nice_increment,
            tf_inter_op_threads=inter_op_threads,
            tf_intra_op_threads=intra_op_threads,
            gpu_strategy=gpu_strategy,
        )

    gpu_memory_limit_mb = configs.get("gpu_memory_limit_mb")
    if gpu_memory_limit_mb:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit_mb)],
            )
            gpu_strategy = f"显存上限 {gpu_memory_limit_mb}MB"
        except RuntimeError:
            gpu_strategy = "显存上限设置失败，沿用默认配置"

    elif configs.get("gpu_memory_growth", True):
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            gpu_strategy = "memory growth"
        except RuntimeError:
            gpu_strategy = "memory growth 设置失败，沿用默认配置"

    return RuntimeSummary(
        device_type="GPU",
        device_names=[gpu.name for gpu in gpus],
        nice_increment=nice_increment,
        tf_inter_op_threads=inter_op_threads,
        tf_intra_op_threads=intra_op_threads,
        gpu_strategy=gpu_strategy,
    )


def build_serving_signature(model, configs):
    image_size = configs["image_size"]

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.string, name="image_bytes"),
        ]
    )
    def serve_bytes(image_bytes):
        def preprocess_single_image(image_content):
            image = tf.io.decode_image(
                image_content, channels=3, expand_animations=False
            )
            image.set_shape([None, None, 3])
            image = tf.image.resize(image, [image_size, image_size])
            image = tf.cast(image, tf.float32) / 255.0
            return image

        batch = tf.map_fn(
            preprocess_single_image,
            image_bytes,
            fn_output_signature=tf.TensorSpec(
                shape=(image_size, image_size, 3), dtype=tf.float32
            ),
        )
        scores = model(batch, training=False)
        class_ids = tf.argmax(scores, axis=1, output_type=tf.int32)
        return {
            "class_ids": class_ids,
            "scores": scores,
        }

    return serve_bytes.get_concrete_function()


def extract_serving_metadata(saved_model_path):
    imported = tf.saved_model.load(saved_model_path)
    serving_fn = imported.signatures["serving_default"]
    input_signature = serving_fn.structured_input_signature[1]
    input_name, input_tensor_spec = next(iter(input_signature.items()))
    structured_outputs = serving_fn.structured_outputs

    return {
        "signature_key": "serving_default",
        "input_key": input_name,
        "input_tensor_name": input_tensor_spec.name,
        "output_tensor_names": {
            output_name: tensor.name
            for output_name, tensor in structured_outputs.items()
        },
    }


def export_serving_artifacts(model, model_name, configs, keras_model_path):
    saved_model_path = os.path.join(configs['saved_model_dir'], model_name)
    if os.path.exists(saved_model_path):
        shutil.rmtree(saved_model_path)

    serving_signature = build_serving_signature(model, configs)
    tf.saved_model.save(
        model,
        saved_model_path,
        signatures={"serving_default": serving_signature},
    )

    manifest = {
        "default_model": model_name,
        "keras_model_path": keras_model_path,
        "saved_model_path": saved_model_path,
        "labels_path": configs['labels_path'],
        "image_size": configs['image_size'],
        "num_classes": configs['num_classes'],
        "serving": extract_serving_metadata(saved_model_path),
    }

    with open(configs['manifest_path'], 'w', encoding='utf-8') as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)

    return manifest


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
    configs["validation_freq"] = int(
        configs.get("validation_frequency", configs.get("validation_freq", 1)) or 1
    )
    refresh_per_second = float(
        configs.get(
            "rich_progress_refresh_per_second",
            configs.get("progress_refresh_seconds", 4),
        )
        or 4
    )
    configs["progress_refresh_seconds"] = 1 / max(refresh_per_second, 1.0)
    return configs


def build_model_wrapper(model_name, image_size, num_classes):
    model_class = MODEL_BUILDERS.get(model_name, ButterflyC)
    return model_class((image_size, image_size, 3), num_classes=num_classes)


def fit_stage(model, dataset_bundle, epochs, callbacks, validation_freq):
    if epochs <= 0:
        return None

    return model.fit(
        dataset_bundle.train_data,
        epochs=epochs,
        validation_data=dataset_bundle.val_data,
        validation_freq=validation_freq,
        callbacks=callbacks,
        verbose=0,
    )


def build_callbacks(configs, monitor, stage_key, stage_name, epochs, dataset_bundle):
    monitor_metric = 'val_loss' if configs['validation_freq'] == 1 else 'loss'
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.8,
        patience=5,
        min_lr=0.00005,
    )
    checkpoint = ModelCheckpoint(
        os.path.join(configs['model_path'], 'checkpoint.keras'),
        monitor=monitor_metric,
        save_best_only=True,
    )
    csv_logger = CSVLogger(
        os.path.join(configs['log_dir'], f'{stage_key}.csv'),
        append=False,
    )
    progress_callback = monitor.build_progress_callback(
        stage_name=stage_name,
        total_epochs=epochs,
        train_steps=dataset_bundle.train_steps,
        refresh_seconds=configs['progress_refresh_seconds'],
    )
    terminate_on_nan = TerminateOnNaN()
    return [
        reduce_lr,
        checkpoint,
        csv_logger,
        progress_callback,
        terminate_on_nan,
    ]


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=False,
    )


def run_training_stage(
    *,
    model,
    dataset_bundle,
    configs,
    monitor,
    stage_key,
    stage_name,
    epochs,
    learning_rate,
):
    if epochs <= 0:
        monitor.log_skip(stage_name, "epoch 数为 0")
        return None

    compile_model(model, learning_rate)
    callbacks = build_callbacks(
        configs,
        monitor,
        stage_key,
        stage_name,
        epochs,
        dataset_bundle,
    )
    monitor.log_stage_start(
        stage_name=stage_name,
        epochs=epochs,
        learning_rate=learning_rate,
        dataset_bundle=dataset_bundle,
        trainable_params=count_trainable_parameters(model),
    )
    stage_started_at = time.perf_counter()
    history = fit_stage(
        model,
        dataset_bundle,
        epochs,
        callbacks,
        configs['validation_freq'],
    )
    monitor.log_stage_end(
        stage_name,
        history,
        model,
        elapsed_seconds=time.perf_counter() - stage_started_at,
    )
    return history


def train(
    model_name='ButterflyC',
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
    runtime_summary = configure_runtime(configs)
    n = configs['num_classes']
    size = configs['image_size']
    resolved_model_name = model_name if model_name in MODEL_BUILDERS else "ButterflyC"
    monitor = TrainingMonitor(configs['log_dir'], resolved_model_name)
    if model_name != resolved_model_name:
        monitor.logger.info(
            "未识别的模型名 %s，回退到 %s。",
            model_name,
            resolved_model_name,
        )

    label_encoder = export_label_artifacts(configs['train_csv'], configs['labels_path'])
    dataset_bundle = load_data(
        configs=configs,
        sample_limit=sample_limit,
        label_encoder=label_encoder,
    )
    butterfly_model = build_model_wrapper(resolved_model_name, size, n)
    model = butterfly_model.build_model()
    monitor.show_training_plan(
        model_name=resolved_model_name,
        configs=configs,
        dataset_bundle=dataset_bundle,
        runtime=runtime_summary,
        sample_limit=sample_limit,
        skip_fine_tuning=skip_fine_tuning,
    )
    if not dataset_bundle.stratified_split:
        monitor.logger.info("当前数据切分未使用分层抽样。")
    if sample_limit is not None:
        monitor.logger.info("当前运行启用了 sample_limit=%s。", sample_limit)

    history = run_training_stage(
        model=model,
        dataset_bundle=dataset_bundle,
        configs=configs,
        monitor=monitor,
        stage_key='initial_training',
        stage_name='初始训练',
        epochs=configs['initial_epochs'],
        learning_rate=configs['learning_rate'],
    )
    init_model_path = os.path.join(
        configs['model_path'],
        f'{resolved_model_name}-init.keras',
    )
    model.save(init_model_path)
    monitor.logger.info("已保存初始阶段模型: %s", init_model_path)

    history_fine = None
    if not skip_fine_tuning and configs["fine_tuning_epochs"] > 0:
        model = butterfly_model.unfreeze_base_model(model)
        history_fine = run_training_stage(
            model=model,
            dataset_bundle=dataset_bundle,
            configs=configs,
            monitor=monitor,
            stage_key='fine_tuning',
            stage_name='微调训练',
            epochs=configs['fine_tuning_epochs'],
            learning_rate=configs['fine_tuning_learning_rate'],
        )
    else:
        monitor.log_skip("微调训练", "命令行跳过或 fine_tuning_epochs 为 0")
    keras_model_path = os.path.join(configs['model_path'],f'{resolved_model_name}.keras')
    model.save(keras_model_path)
    monitor.logger.info("已保存最终 Keras 模型: %s", keras_model_path)
    manifest = export_serving_artifacts(model, resolved_model_name, configs, keras_model_path)
    monitor.log_artifacts(
        keras_model_path=keras_model_path,
        init_model_path=init_model_path,
        manifest=manifest,
        manifest_path=configs['manifest_path'],
    )
    monitor.logger.info("训练完成，SavedModel 与清单文件已导出。")
    return history, history_fine


def parse_args():
    parser = argparse.ArgumentParser(description="Train ButterflyC models.")
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
