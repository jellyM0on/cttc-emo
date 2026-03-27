from __future__ import annotations

from pathlib import Path

import argparse
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

from src.config import TrainingConfig
from src.data_utils import prepare_data
from src.models.attention import AttentionPooling
from eval.eval_metrics import (
    compute_overall_metrics,
    compute_per_label_metrics,
    compute_truth_table_counts,
)
from eval.eval_plots import (
    plot_per_label_f1,
    plot_truth_table_chart,
)
from eval.eval_utils import save_json, save_csv, save_text

def predict_probabilities(model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    y_prob = model.predict(dataset, verbose=0)
    return np.asarray(y_prob)

def evaluate_split(
    split_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    threshold: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, np.ndarray]:
    overall_metrics, y_pred = compute_overall_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
    )

    per_label_df = compute_per_label_metrics(
        y_true=y_true,
        y_prob=y_prob,
        label_names=label_names,
        threshold=threshold,
    )

    truth_table_df = compute_truth_table_counts(
        y_true=y_true,
        y_prob=y_prob,
        label_names=label_names,
        threshold=threshold,
    )

    overall_metrics["split"] = split_name
    return overall_metrics, per_label_df, truth_table_df, y_pred


def load_model(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"AttentionPooling": AttentionPooling},
    )

def run_evaluation(
    config: TrainingConfig,
) -> dict:
    """
    Evaluates train/validation/ test using the saved model and dataset.

    """
    config.ensure_dirs()

    # Output folders
    metrics_dir = config.eval_outputs_dir / "metrics" / config.model_name
    reports_dir = config.eval_outputs_dir / "reports" / config.model_name
    figures_dir = config.eval_outputs_dir/ "figures" / config.model_name

    data = prepare_data(config)
    model = load_model(config.model_output_path)

    split_payload = {
        "train": (data["train_ds"], data["y_train"]),
        "validation": (data["val_ds"], data["y_val"]),
        "test": (data["test_ds"], data["y_test"]),
    }

    overall_rows = []
    results = {}

    for split_name, (dataset, y_true) in split_payload.items():
        y_prob = predict_probabilities(model, dataset)

        overall_metrics, per_label_df, truth_table_df, y_pred = evaluate_split(
            split_name=split_name,
            y_true=y_true,
            y_prob=y_prob,
            label_names=data["label_cols"],
            threshold=config.threshold,
        )

        overall_rows.append(overall_metrics)

        save_json(overall_metrics, metrics_dir / f"{split_name}_overall_metrics.json")
        save_csv(per_label_df, reports_dir / f"{split_name}_per_label_f1_support.csv")
        save_csv(truth_table_df, reports_dir / f"{split_name}_truth_table_counts.csv")

        cls_report = classification_report(
            y_true,
            y_pred,
            target_names=data["label_cols"],
            zero_division=0,
        )
        save_text(cls_report, reports_dir / f"{split_name}_classification_report.txt")

        plot_per_label_f1(
            per_label_df=per_label_df,
            model_name=f"{config.model_name}_{split_name}",
            save_path=figures_dir / f"{split_name}_per_label_f1.png",
        )

        plot_truth_table_chart(
            truth_table_df=truth_table_df,
            model_name=f"{config.model_name}_{split_name}",
            save_path=figures_dir / f"{split_name}_truth_table.png",
        )
            
        results[split_name] = {
            "overall_metrics": overall_metrics,
            "per_label_df": per_label_df,
            "truth_table_df": truth_table_df,
        }

    overall_df = pd.DataFrame(overall_rows)[
        ["split", "accuracy", "precision", "recall", "micro_f1", "macro_f1"]
    ]
    save_csv(overall_df, metrics_dir / "all_splits_overall_metrics.csv")

    return results

VALID_MODELS = ["baseline", "attention", "stacked"]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one or more GoEmotions models.")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model to evaluate: baseline, attention, or stacked. If omitted, evaluates all models in order.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.model_name is None:
        model_names = VALID_MODELS
    else:
        model_name = args.model_name.strip().lower()

        if model_name not in VALID_MODELS:
            print(
                f"Invalid model_name: {args.model_name}\n"
                f"Valid model values are: {', '.join(VALID_MODELS)}"
            )
            sys.exit(1)

        model_names = [model_name]

    for model_name in model_names:
        print(f"\n=== Evaluating: {model_name} ===")
        config = TrainingConfig(model_name=model_name)
        run_evaluation(config=config)

if __name__ == "__main__":
    main()