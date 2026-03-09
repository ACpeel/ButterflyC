import json
import os

import pandas as pd


class SimpleLabelEncoder:
    def __init__(self, classes):
        if not classes:
            raise ValueError("Label classes cannot be empty.")

        self.classes_ = list(classes)
        self.class_to_index = {
            label: index for index, label in enumerate(self.classes_)
        }

    def transform(self, labels):
        return [self.class_to_index[label] for label in labels]

    def inverse_transform(self, indices):
        return [self.classes_[index] for index in indices]


def build_label_encoder(csv_path):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise KeyError(f"Missing 'label' column in {csv_path}")

    classes = sorted(df["label"].dropna().unique().tolist())
    return SimpleLabelEncoder(classes)


def export_label_artifacts(csv_path, output_path):
    label_encoder = build_label_encoder(csv_path)
    classes = list(label_encoder.classes_)
    payload = {
        "classes": classes,
        "label_to_index": {label: index for index, label in enumerate(classes)},
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as mapping_file:
        json.dump(payload, mapping_file, ensure_ascii=False, indent=2)

    return label_encoder


def load_label_artifacts(labels_path):
    with open(labels_path, "r", encoding="utf-8") as mapping_file:
        payload = json.load(mapping_file)

    classes = payload.get("classes", [])
    if not classes:
        raise ValueError(f"No classes found in {labels_path}")

    return classes
