import json
import os
from functools import lru_cache

import numpy as np
import torch

import main.utils.config as config
from main.torch_model import build_torch_model, normalize_model_name
from main.utils.labels import load_label_artifacts
from main.utils.torch_process import load_inference_tensor

configs = config.load_config()


def load_serving_manifest():
    manifest_path = configs.get("manifest_path")
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            return json.load(manifest_file)
    return {}


class TensorFlowRecognitionService:
    def __init__(self, manifest):
        import tensorflow as tf

        import main.utils.process as tf_process

        keras_model_path = manifest.get(
            "keras_model_path",
            os.path.join(configs["model_path"], "ButterflyC.keras"),
        )
        labels_path = manifest.get("labels_path", configs["labels_path"])

        if not os.path.exists(keras_model_path):
            raise FileNotFoundError(f"Model file not found: {keras_model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        self.tf = tf
        self.tf_process = tf_process
        self.model = tf.keras.models.load_model(keras_model_path)
        self.labels = load_label_artifacts(labels_path)

    def predict(self, pic_path):
        if not pic_path:
            return [None, ["default"]]

        pic_array = self.tf_process.process_img(
            pic_path,
            (configs["image_size"], configs["image_size"]),
        )
        predictions = self.model.predict(pic_array, verbose=0)
        predicted_index = int(np.argmax(predictions, axis=1)[0])

        if predicted_index >= len(self.labels):
            raise IndexError(
                f"Predicted class index {predicted_index} is out of range for labels."
            )

        return [pic_path, [self.labels[predicted_index]]]


class TorchRecognitionService:
    def __init__(self, manifest):
        model_path = manifest.get(
            "torch_model_path",
            os.path.join(configs["model_path"], "ButterflyC.pt"),
        )
        labels_path = manifest.get("labels_path", configs["labels_path"])

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        payload = torch.load(model_path, map_location="cpu")
        model_name = normalize_model_name(
            payload.get("model_name", manifest.get("default_model", "ButterflyC"))
        )
        labels = payload.get("class_names") or load_label_artifacts(labels_path)

        _, model, _ = build_torch_model(
            model_name,
            num_classes=len(labels),
            pretrained=False,
        )
        state_dict = payload.get("state_dict", payload)
        model.load_state_dict(state_dict)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.labels = labels
        self.image_size = int(payload.get("image_size", configs["image_size"]))

    def predict(self, pic_path):
        if not pic_path:
            return [None, ["default"]]

        image_tensor = load_inference_tensor(pic_path, self.image_size).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            predicted_index = int(outputs.argmax(dim=1).item())

        if predicted_index >= len(self.labels):
            raise IndexError(
                f"Predicted class index {predicted_index} is out of range for labels."
            )

        return [pic_path, [self.labels[predicted_index]]]


def build_recognition_service():
    manifest = load_serving_manifest()
    backend = manifest.get("backend")

    if backend == "torch":
        return TorchRecognitionService(manifest)
    if backend == "tensorflow":
        return TensorFlowRecognitionService(manifest)

    torch_model_path = os.path.join(configs["model_path"], "ButterflyC.pt")
    if os.path.exists(torch_model_path):
        return TorchRecognitionService(manifest)
    return TensorFlowRecognitionService(manifest)


@lru_cache(maxsize=1)
def get_recognition_service():
    return build_recognition_service()


def recognize(pic_path):
    result = get_recognition_service().predict(pic_path)
    print(f"{result[0]} ---- {result[1]}")
    return result


if __name__ == "__main__":
    path = "data/test/Image_10.jpg"
    recognize(path)
