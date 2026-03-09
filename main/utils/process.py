import os
from dataclasses import dataclass
from math import ceil

import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from main.utils.config import load_config
from main.utils.labels import build_label_encoder


@dataclass(frozen=True)
class DatasetBundle:
    train_data: tf.data.Dataset
    val_data: tf.data.Dataset
    train_samples: int
    val_samples: int
    train_steps: int
    val_steps: int
    batch_size: int
    class_names: tuple[str, ...]
    stratified_split: bool

    @property
    def train_count(self):
        return self.train_samples

    @property
    def val_count(self):
        return self.val_samples

    @property
    def total_count(self):
        return self.train_samples + self.val_samples


def process_img(img_path, target_size):
    image_bytes = tf.io.read_file(img_path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image.numpy()


def apply_smote(train_images, train_labels):
    try:
        configs = load_config()
        smote = SMOTE()
        print("Applying SMOTE...")
        train_images_flat = train_images.reshape(len(train_images), -1)
        train_images_resampled, train_labels_resampled = smote.fit_resample(train_images_flat, train_labels)
        return train_images_resampled.reshape(-1, configs['image_size'], configs['image_size'], 3), train_labels_resampled
    except Exception as e:
        print(f"Error applying SMOTE: {e}")
        return train_images, train_labels


def build_data_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])


def load_dataset_item(image_path, label, image_size, num_classes):
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=num_classes)
    return image, label


def build_tf_dataset(
    image_paths,
    labels,
    configs,
    *,
    augment=False,
    shuffle=False,
):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    parallel_calls = configs.get('tf_data_parallel_calls', 1)

    if shuffle and image_paths:
        dataset = dataset.shuffle(
            min(len(image_paths), configs['tf_shuffle_buffer']),
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda path, label: load_dataset_item(
            path,
            label,
            configs['image_size'],
            configs['num_classes'],
        ),
        num_parallel_calls=parallel_calls,
    )
    dataset = dataset.batch(configs['batch_size'])

    if augment:
        data_augmentation = build_data_augmentation()
        dataset = dataset.map(
            lambda images, labels: (data_augmentation(images, training=True), labels),
            num_parallel_calls=1,
        )

    options = tf.data.Options()
    options.threading.private_threadpool_size = configs['tf_data_private_threadpool_size']
    options.threading.max_intra_op_parallelism = configs['tf_data_max_intra_op_parallelism']
    dataset = dataset.with_options(options)
    dataset = dataset.prefetch(configs['tf_data_prefetch'])
    return dataset


def build_training_dataframe(
    configs,
    sample_limit=None,
    random_state=42,
    label_encoder=None,
):
    train_df = pd.read_csv(configs['train_csv'])
    label_encoder = label_encoder or encode_labels(configs['train_csv'])

    if sample_limit is not None and sample_limit > 0 and len(train_df) > sample_limit:
        train_df = train_df.sample(n=sample_limit, random_state=random_state)
        train_df = train_df.reset_index(drop=True)

    train_df['label_id'] = label_encoder.transform(train_df['label'])
    train_df['image_path'] = train_df['filename'].map(
        lambda filename: os.path.join(configs['train_data'], str(filename))
    )
    return train_df, label_encoder


def should_stratify(label_ids):
    if len(label_ids) < 2:
        return False

    counts = pd.Series(label_ids).value_counts()
    return bool(not counts.empty and counts.min() >= 2)


def split_dataset_entries(train_df, *, random_state, val_split):
    label_ids = train_df['label_id'].tolist()
    stratify = label_ids if should_stratify(label_ids) else None
    return train_test_split(
        train_df['image_path'].tolist(),
        label_ids,
        test_size=val_split,
        random_state=random_state,
        stratify=stratify,
    )

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
    train_data = build_tf_dataset(
        train_paths,
        train_labels,
        configs,
        augment=True,
        shuffle=True,
    )
    val_data = build_tf_dataset(
        val_paths,
        val_labels,
        configs,
        augment=False,
        shuffle=False,
    )
    batch_size = max(1, int(configs['batch_size']))
    stratified_split = should_stratify(train_df['label_id'].tolist())
    return DatasetBundle(
        train_data=train_data,
        val_data=val_data,
        train_samples=len(train_paths),
        val_samples=len(val_paths),
        train_steps=ceil(len(train_paths) / batch_size),
        val_steps=ceil(len(val_paths) / batch_size),
        batch_size=batch_size,
        class_names=tuple(label_encoder.classes_),
        stratified_split=stratified_split,
    )

def encode_labels(csv_path):
    return build_label_encoder(csv_path)
