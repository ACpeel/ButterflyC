import json
import os
from functools import lru_cache

import numpy as np
import tensorflow as tf

import main.utils.config as config
import main.utils.process as process
from main.utils.labels import load_label_artifacts

configs = config.load_config()

def load_serving_manifest():
    manifest_path = configs.get("manifest_path")
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            return json.load(manifest_file)
    return {}


class RecognitionService:
    def __init__(self):
        manifest = load_serving_manifest()
        keras_model_path = manifest.get(
            "keras_model_path",
            os.path.join(configs["model_path"], "ButterflyC.keras"),
        )
        labels_path = manifest.get("labels_path", configs["labels_path"])

        if not os.path.exists(keras_model_path):
            raise FileNotFoundError(f"Model file not found: {keras_model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        self.model = tf.keras.models.load_model(keras_model_path)
        self.labels = load_label_artifacts(labels_path)

    def predict(self, pic_path):
        if not pic_path:
            return [None, ["default"]]

        pic_array = process.process_img(
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


@lru_cache(maxsize=1)
def get_recognition_service():
    return RecognitionService()


def recognize(pic_path):
    result = get_recognition_service().predict(pic_path)
    print(f"{result[0]} ---- {result[1]}")
    return result


if __name__ == "__main__":
    path = "data/test/Image_10.jpg"
    recognize(path)
