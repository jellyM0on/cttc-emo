import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score

from config import TrainingConfig
from data_utils import prepare_data
from pathlib import Path
from models.baseline import build_baseline_model
from models.attention import build_attention_model
from models.stacked import build_stacked_model

def compile_model(model):
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

def get_callbacks(config: TrainingConfig):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        )
    ]

def evaluate_model(model, test_ds, y_test, threshold: float):
    test_loss, test_binary_acc, test_precision, test_recall = model.evaluate(test_ds)
    y_pred_probs = model.predict(test_ds)
    y_pred = (y_pred_probs >= threshold).astype(np.int32)
    y_true = y_test.astype(np.int32)

    metrics = {
        "test_loss": test_loss,
        "test_binary_accuracy": test_binary_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    return metrics

def build_model(config: TrainingConfig, vectorizer, num_classes: int):
    if config.model_name == "baseline":
        return build_baseline_model(
            vectorizer=vectorizer,
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            lstm_units=config.lstm_units,
            dropout_rate=config.dropout_rate,
            num_classes=num_classes,
        )
    if config.model_name == "attention":
        return build_attention_model(
            vectorizer=vectorizer,
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            lstm_units=config.lstm_units,
            dropout_rate=config.dropout_rate,
            num_classes=num_classes,
        )
    if config.model_name == "stacked":
        return build_stacked_model(
            vectorizer=vectorizer,
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            lstm_units=config.lstm_units,
            dropout_rate=config.dropout_rate,
            num_classes=num_classes,
        )
    raise ValueError(f"Unsupported model_name: {config.model_name}")

def save_training_results(history, model_name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["binary_accuracy"], label="Train Binary Accuracy")
    plt.plot(history.history["val_binary_accuracy"], label="Validation Binary Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Accuracy")
    plt.title(f"Training vs Validation Binary Accuracy ({model_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_binary_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss ({model_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

def run_experiment(config: TrainingConfig):
    config.ensure_dirs()
    data = prepare_data(config)

    model = build_model(config, data["vectorizer"], data["num_classes"])
    model = compile_model(model)
    model.summary()

    history = model.fit(
        data["train_ds"],
        validation_data=data["val_ds"],
        epochs=config.epochs,
        callbacks=get_callbacks(config),
    )

    metrics = evaluate_model(
        model,
        data["test_ds"],
        data["y_test"],
        threshold=config.threshold,
    )

    model.save(str(config.model_output_path))
    save_training_results(history, config.model_name, config.figures_dir)

    print("\nTest Metrics")
    print(f"Test Loss:            {metrics['test_loss']:.4f}")
    print(f"Test Binary Accuracy: {metrics['test_binary_accuracy']:.4f}")
    print(f"Test Precision:       {metrics['test_precision']:.4f}")
    print(f"Test Recall:          {metrics['test_recall']:.4f}")
    print(f"F1 Micro:             {metrics['f1_micro']:.4f}")
    print(f"F1 Macro:             {metrics['f1_macro']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])

    return model, history, metrics, data
