import json
import os
import shutil

import tensorflow as tf


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
                image_content,
                channels=3,
                expand_animations=False,
            )
            image.set_shape([None, None, 3])
            image = tf.image.resize(image, [image_size, image_size])
            image = tf.cast(image, tf.float32) / 255.0
            return image

        batch = tf.map_fn(
            preprocess_single_image,
            image_bytes,
            fn_output_signature=tf.TensorSpec(
                shape=(image_size, image_size, 3),
                dtype=tf.float32,
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
    saved_model_path = os.path.join(configs["saved_model_dir"], model_name)
    if os.path.exists(saved_model_path):
        shutil.rmtree(saved_model_path)

    serving_signature = build_serving_signature(model, configs)
    tf.saved_model.save(
        model,
        saved_model_path,
        signatures={"serving_default": serving_signature},
    )

    manifest = {
        "backend": "tensorflow",
        "default_model": model_name,
        "keras_model_path": keras_model_path,
        "saved_model_path": saved_model_path,
        "labels_path": configs["labels_path"],
        "image_size": configs["image_size"],
        "num_classes": configs["num_classes"],
        "serving": extract_serving_metadata(saved_model_path),
    }

    with open(configs["manifest_path"], "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)

    return manifest
