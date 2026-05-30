#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
FINE_TUNE_AT = 100

CLASS_LABELS_PATH = Path("model/class_labels.json")
MODEL_PATH = Path("model/plantmd_model.keras")
LOG_DIR = Path("logs/fit") / str(int(time.time()))

# ─────────────────────────────────────────────────────────────
# Dataset Loader
# ─────────────────────────────────────────────────────────────

def build_datasets(data_dir: str):

    data_dir = Path(data_dir)

    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"

    assert train_dir.exists(), f"Train directory not found: {train_dir}"
    assert valid_dir.exists(), f"Validation directory not found: {valid_dir}"

    train_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.20,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255
    )

    train_ds = train_gen.flow_from_directory(
        directory=str(train_dir),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )

    val_ds = val_gen.flow_from_directory(
        directory=str(valid_dir),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    class_names = [
        k for k, v in sorted(train_ds.class_indices.items(), key=lambda x: x[1])
    ]

    CLASS_LABELS_PATH.write_text(json.dumps(class_names, indent=2))

    print(f"\n✓ Found {train_ds.num_classes} classes")
    print(f"✓ Training images: {train_ds.samples}")
    print(f"✓ Validation images: {val_ds.samples}")

    return train_ds, val_ds, class_names


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

def build_model(num_classes: int):

    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="PlantMD_MobileNetV2")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(
                k=3,
                name="top3_acc"
            ),
        ],
    )

    model.summary()

    return model


# ─────────────────────────────────────────────────────────────
# Fine Tuning
# ─────────────────────────────────────────────────────────────

def enable_fine_tuning(model):

    base = model.layers[1]

    base.trainable = True

    for layer in base.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(
                k=3,
                name="top3_acc"
            ),
        ],
    )

    print(f"\n✓ Fine-tuning enabled from layer {FINE_TUNE_AT}")


# ─────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────

def get_callbacks(phase: str):

    return [
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),

        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),

        TensorBoard(
            log_dir=str(LOG_DIR / phase),
            histogram_freq=1,
        ),
    ]


# ─────────────────────────────────────────────────────────────
# Training Pipeline
# ─────────────────────────────────────────────────────────────

def train(data_dir: str, epochs: int, batch_size: int):

    global BATCH_SIZE, EPOCHS

    BATCH_SIZE = batch_size
    EPOCHS = epochs

    # GPU memory growth
    gpus = tf.config.experimental.list_physical_devices("GPU")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    strategy = tf.distribute.MirroredStrategy()

    print(f"\n✓ Using {strategy.num_replicas_in_sync} GPU(s)")

    train_ds, val_ds, class_names = build_datasets(data_dir)

    num_classes = len(class_names)

    with strategy.scope():
        model = build_model(num_classes)

    # Phase 1
    print("\n── Phase 1: Training classification head ──")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(1, epochs // 2),
        callbacks=get_callbacks("phase1"),
        verbose=1,
    )

    # Phase 2
    print("\n── Phase 2: Fine-tuning backbone ──")

    enable_fine_tuning(model)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=len(history1.epoch),
        callbacks=get_callbacks("phase2"),
        verbose=1,
    )

    # Evaluation
    print("\n── Final Evaluation ──")

    loss, acc, top3 = model.evaluate(val_ds, verbose=1)

    print(f"\nValidation Accuracy: {acc:.4f}")
    print(f"Top-3 Accuracy: {top3:.4f}")

    # Save model
    model.save(str(MODEL_PATH))

    print(f"\n✓ Model saved → {MODEL_PATH}")
    print(f"✓ Class labels saved → {CLASS_LABELS_PATH}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train PlantMD classifier"
    )

    parser.add_argument(
        "--data",
        default="./PlantVillage",
        help="Path to dataset",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size",
    )

    args = parser.parse_args()

    train(
        args.data,
        args.epochs,
        args.batch,
    )