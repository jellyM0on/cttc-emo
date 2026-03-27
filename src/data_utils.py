from pathlib import Path

import numpy as np
import tensorflow as tf
from datasets import load_dataset, load_from_disk

AUTOTUNE = tf.data.AUTOTUNE

def get_goemotions_simplified(local_path: Path):
    if local_path.exists():
        print(f"Loading dataset from local cache: {local_path}")
        return load_from_disk(str(local_path))

    print("Local dataset not found. Downloading from Hugging Face...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    ds.save_to_disk(str(local_path))
    print(f"Dataset saved locally to: {local_path}")
    return ds

def to_multihot(label_lists, num_classes: int) -> np.ndarray:
    y = np.zeros((len(label_lists), num_classes), dtype=np.float32)
    for i, labels in enumerate(label_lists):
        if labels:
            y[i, labels] = 1.0
    return y

def extract_splits_and_labels(ds):
    train_split = ds["train"]
    val_split = ds["validation"]
    test_split = ds["test"]

    label_feature = train_split.features["labels"].feature
    label_cols = label_feature.names
    num_classes = len(label_cols)

    x_train = [str(x) for x in train_split["text"]]
    x_val = [str(x) for x in val_split["text"]]
    x_test = [str(x) for x in test_split["text"]]

    y_train = to_multihot(train_split["labels"], num_classes)
    y_val = to_multihot(val_split["labels"], num_classes)
    y_test = to_multihot(test_split["labels"], num_classes)

    return {
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "label_cols": label_cols,
        "num_classes": num_classes,
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

def build_vectorizer(x_train, vocab_size: int, sequence_length: int, batch_size: int):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    train_text_ds = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)
    vectorizer.adapt(train_text_ds)
    return vectorizer

def make_dataset(x, y, batch_size: int, shuffle: bool = False, cache: bool = True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x))
    ds = ds.batch(batch_size)
    if cache:
        ds = ds.cache()
    return ds.prefetch(AUTOTUNE)

def prepare_data(config):
    ds = get_goemotions_simplified(config.hf_cache_dir)
    data = extract_splits_and_labels(ds)
    vectorizer = build_vectorizer(
        data["x_train"],
        vocab_size=config.vocab_size,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
    )

    train_ds = make_dataset(data["x_train"], data["y_train"], config.batch_size, shuffle=True)
    val_ds = make_dataset(data["x_val"], data["y_val"], config.batch_size)
    test_ds = make_dataset(data["x_test"], data["y_test"], config.batch_size)

    data["vectorizer"] = vectorizer
    data["train_ds"] = train_ds
    data["val_ds"] = val_ds
    data["test_ds"] = test_ds
    return data
