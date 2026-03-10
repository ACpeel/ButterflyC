import os
from dataclasses import dataclass
from math import ceil

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from main.utils.config import load_config
from main.utils.labels import build_label_encoder


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_samples: int
    val_samples: int
    train_steps: int
    val_steps: int
    batch_size: int
    class_names: tuple[str, ...]
    stratified_split: bool


class ButterflyTorchDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label


def build_train_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_inference_tensor(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    transform = build_eval_transform(image_size)
    return transform(image).unsqueeze(0)


def build_training_dataframe(
    configs,
    sample_limit=None,
    random_state=42,
    label_encoder=None,
):
    train_df = pd.read_csv(configs["train_csv"])
    label_encoder = label_encoder or encode_labels(configs["train_csv"])

    if sample_limit is not None and sample_limit > 0 and len(train_df) > sample_limit:
        train_df = train_df.sample(n=sample_limit, random_state=random_state)
        train_df = train_df.reset_index(drop=True)

    train_df["label_id"] = label_encoder.transform(train_df["label"])
    train_df["image_path"] = train_df["filename"].map(
        lambda filename: os.path.join(configs["train_data"], str(filename))
    )
    return train_df, label_encoder


def should_stratify(label_ids):
    if len(label_ids) < 2:
        return False

    counts = pd.Series(label_ids).value_counts()
    return bool(not counts.empty and counts.min() >= 2)


def split_dataset_entries(train_df, *, random_state, val_split):
    label_ids = train_df["label_id"].tolist()
    stratify = label_ids if should_stratify(label_ids) else None
    return train_test_split(
        train_df["image_path"].tolist(),
        label_ids,
        test_size=val_split,
        random_state=random_state,
        stratify=stratify,
    )


def build_dataloader(dataset, configs, *, shuffle):
    num_workers = max(0, int(configs.get("torch_num_workers", 0) or 0))
    persistent_workers = bool(configs.get("torch_persistent_workers", False))
    pin_memory = bool(configs.get("torch_pin_memory", True))

    loader_kwargs = {
        "batch_size": int(configs["batch_size"]),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        prefetch_factor = configs.get("torch_prefetch_factor")
        if prefetch_factor:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(dataset, **loader_kwargs)


def load_data(
    configs=None,
    sample_limit=None,
    random_state=42,
    val_split=0.1,
    label_encoder=None,
):
    configs = configs or load_config()
    train_df, label_encoder = build_training_dataframe(
        configs,
        sample_limit=sample_limit,
        random_state=random_state,
        label_encoder=label_encoder,
    )
    train_paths, val_paths, train_labels, val_labels = split_dataset_entries(
        train_df,
        random_state=random_state,
        val_split=val_split,
    )

    train_dataset = ButterflyTorchDataset(
        train_paths,
        train_labels,
        build_train_transform(configs["image_size"]),
    )
    val_dataset = ButterflyTorchDataset(
        val_paths,
        val_labels,
        build_eval_transform(configs["image_size"]),
    )

    batch_size = max(1, int(configs["batch_size"]))
    return DatasetBundle(
        train_loader=build_dataloader(train_dataset, configs, shuffle=True),
        val_loader=build_dataloader(val_dataset, configs, shuffle=False),
        train_samples=len(train_paths),
        val_samples=len(val_paths),
        train_steps=ceil(len(train_paths) / batch_size),
        val_steps=ceil(len(val_paths) / batch_size),
        batch_size=batch_size,
        class_names=tuple(label_encoder.classes_),
        stratified_split=should_stratify(train_df["label_id"].tolist()),
    )


def encode_labels(csv_path):
    return build_label_encoder(csv_path)


def move_batch_to_device(images, labels, device, *, channels_last=False):
    images = images.to(device, non_blocking=True)
    if channels_last and images.dim() == 4:
        images = images.contiguous(memory_format=torch.channels_last)
    labels = labels.to(device, non_blocking=True)
    return images, labels
