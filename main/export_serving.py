import argparse
import os

import tensorflow as tf

from main.train import export_serving_artifacts
from main.utils.config import load_config
from main.utils.labels import export_label_artifacts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export SavedModel and label metadata for C++ inference."
    )
    parser.add_argument(
        "--model-name",
        default="ButterflyC",
        help="Keras model name without extension, default: ButterflyC",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Optional absolute or relative path to an existing .keras model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configs = load_config()

    keras_model_path = args.model_path or os.path.join(
        configs["model_path"], f"{args.model_name}.keras"
    )
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Model file not found: {keras_model_path}")

    export_label_artifacts(configs["train_csv"], configs["labels_path"])
    model = tf.keras.models.load_model(keras_model_path)
    export_serving_artifacts(model, args.model_name, configs, keras_model_path)
    print(f"Exported SavedModel to {os.path.join(configs['saved_model_dir'], args.model_name)}")
    print(f"Updated manifest: {configs['manifest_path']}")


if __name__ == "__main__":
    main()
