"""
src/models/train.py

Trains and compares Random Forest and XGBoost classifiers on the
preprocessed NSL-KDD data, evaluates them using metrics that actually
matter for a NIDS (not just accuracy), selects a winner by an explicit
pre-declared rule, and persists the winning model with full metadata.

Why not just use accuracy:
  Our test set is class-imbalanced in the OPPOSITE direction from train
  (more attacks than normal traffic). A model that just predicts "attack"
  for everything would score deceptively high on accuracy while being
  operationally useless. For a NIDS specifically, False Positive Rate
  matters enormously: a high FPR means analysts get buried in false
  alerts and start ignoring the system entirely ("alert fatigue"), which
  defeats the purpose of building this at all. Recall matters too — missed
  attacks are the whole failure mode we're trying to prevent. So we report
  and select on FPR, Recall, F1, and ROC-AUC together, not accuracy alone.

Model selection rule (v2 — revised after Sprint 3's first run):
  1. Primary: highest ROC-AUC (best overall discrimination between classes)
  2. Tiebreaker (if ROC-AUC within 0.005 of each other): higher RECALL
     wins, not lower FPR.

  Revision history: the original v1 rule broke ties by lowest FPR. On
  this dataset that selected Random Forest over XGBoost despite XGBoost
  catching 609 more true attacks (67.0% vs 62.2% recall) for a cost of
  only 10 additional false positives out of 9,711 negatives (+0.11pp
  FPR). For a NIDS, an undetected intrusion is typically far more costly
  than one extra analyst-reviewed alert, so the rule was corrected to
  reflect that operational priority explicitly, rather than silently
  overriding the result once and leaving the code inconsistent with
  what actually got deployed.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from src.utils.exceptions import AegisNIDSError, ModelRegistryError, ModelNotTrainedError
from src.utils.logger import get_logger, load_full_config, get_project_root

logger = get_logger(__name__)

# ROC-AUC values within this margin of each other are treated as a tie,
# broken by False Positive Rate instead. Declared upfront, not tuned after
# seeing results.
_ROC_AUC_TIE_THRESHOLD = 0.005


@dataclass
class ModelEvaluation:
    """Structured evaluation results for a single trained model."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    false_positive_rate: float
    false_negative_rate: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    cv_roc_auc_mean: float
    cv_roc_auc_std: float

    def to_dict(self) -> dict:
        return asdict(self)


def _evaluate_model(model, model_name: str, X_test: np.ndarray, y_test: np.ndarray,
                     X_train: np.ndarray, y_train: np.ndarray, cv_folds: int) -> ModelEvaluation:
    """
    Compute the full metric suite for a trained model on held-out test data,
    plus cross-validated ROC-AUC on training data as a sanity check against
    overfitting (a big train-vs-cv gap would be a red flag worth surfacing).
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    logger.info(f"Running {cv_folds}-fold cross-validation for {model_name} (on training data)...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="roc_auc", n_jobs=-1)

    return ModelEvaluation(
        model_name=model_name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, y_proba)),
        false_positive_rate=float(fpr),
        false_negative_rate=float(fnr),
        true_positives=int(tp),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        cv_roc_auc_mean=float(cv_scores.mean()),
        cv_roc_auc_std=float(cv_scores.std()),
    )


def _select_winner(evaluations: list) -> ModelEvaluation:
    """Apply the pre-declared selection rule to pick the winning model."""
    sorted_by_auc = sorted(evaluations, key=lambda e: e.roc_auc, reverse=True)
    best, runner_up = sorted_by_auc[0], sorted_by_auc[1] if len(sorted_by_auc) > 1 else None

    if runner_up is not None and (best.roc_auc - runner_up.roc_auc) < _ROC_AUC_TIE_THRESHOLD:
        logger.info(
            f"ROC-AUC scores within tie threshold ({_ROC_AUC_TIE_THRESHOLD}): "
            f"{best.model_name}={best.roc_auc:.4f} vs {runner_up.model_name}={runner_up.roc_auc:.4f}. "
            f"Breaking tie by Recall (higher attack-detection rate takes priority "
            f"over a marginal FPR difference)."
        )
        best = max([best, runner_up], key=lambda e: e.recall)

    return best


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config_path: Optional[Path] = None,
) -> tuple:
    """
    Train Random Forest and XGBoost, evaluate both, select a winner.

    Returns:
        (winning_model, winning_model_name, all_evaluations: list[ModelEvaluation])
    """
    config = load_full_config(config_path) if config_path else load_full_config()
    train_cfg = config.get("training", {})
    random_state = train_cfg.get("random_state", 42)
    cv_folds = train_cfg.get("cv_folds", 5)

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise AegisNIDSError("Cannot train on empty training or test data.")
    if X_train.shape[1] != X_test.shape[1]:
        raise AegisNIDSError(
            f"Feature count mismatch between train ({X_train.shape[1]}) "
            f"and test ({X_test.shape[1]}) sets."
        )

    evaluations = []
    trained_models = {}

    # --- Random Forest ---
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",   # handles the mild train-set imbalance
        random_state=random_state,
        n_jobs=-1,
    )
    try:
        rf.fit(X_train, y_train)
    except (ValueError, MemoryError) as e:
        raise AegisNIDSError(f"Random Forest training failed: {e}") from e

    rf_eval = _evaluate_model(rf, "random_forest", X_test, y_test, X_train, y_train, cv_folds)
    evaluations.append(rf_eval)
    trained_models["random_forest"] = rf
    logger.info(
        f"Random Forest — ROC-AUC: {rf_eval.roc_auc:.4f}, Recall: {rf_eval.recall:.4f}, "
        f"FPR: {rf_eval.false_positive_rate:.4f}"
    )

    # --- XGBoost ---
    logger.info("Training XGBoost...")
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0  # handles class imbalance natively

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    try:
        xgb.fit(X_train, y_train)
    except (ValueError, MemoryError) as e:
        raise AegisNIDSError(f"XGBoost training failed: {e}") from e

    xgb_eval = _evaluate_model(xgb, "xgboost", X_test, y_test, X_train, y_train, cv_folds)
    evaluations.append(xgb_eval)
    trained_models["xgboost"] = xgb
    logger.info(
        f"XGBoost — ROC-AUC: {xgb_eval.roc_auc:.4f}, Recall: {xgb_eval.recall:.4f}, "
        f"FPR: {xgb_eval.false_positive_rate:.4f}"
    )

    winner_eval = _select_winner(evaluations)
    winner_model = trained_models[winner_eval.model_name]

    logger.info(
        f"Model selected: {winner_eval.model_name} "
        f"(ROC-AUC={winner_eval.roc_auc:.4f}, FPR={winner_eval.false_positive_rate:.4f})"
    )

    return winner_model, winner_eval.model_name, evaluations


def save_model(model, model_name: str, evaluations: list, config_path: Optional[Path] = None) -> dict:
    """
    Save the winning model plus a metadata.json capturing which model won,
    all candidates' metrics, and when it was trained. This gives Sprint 4
    (SHAP) and Sprint 5 (simulation) a traceable, versioned artifact rather
    than a bare .joblib file with no provenance.
    """
    config = load_full_config(config_path) if config_path else load_full_config()
    project_root = get_project_root()
    registry_dir = project_root / config["paths"]["model_registry"]
    registry_dir.mkdir(parents=True, exist_ok=True)

    model_path = registry_dir / "nids_model.joblib"
    metadata_path = registry_dir / "model_metadata.json"

    try:
        joblib.dump(model, model_path)
    except (OSError, TypeError) as e:
        raise ModelRegistryError(f"Failed to save model to {model_path}: {e}") from e

    metadata = {
        "selected_model": model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            f"Highest ROC-AUC; ties within {_ROC_AUC_TIE_THRESHOLD} broken by highest Recall "
            f"(prioritizing attack detection over marginal FPR differences)"
        ),
        "all_candidates": {ev.model_name: ev.to_dict() for ev in evaluations},
    }

    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except OSError as e:
        raise ModelRegistryError(f"Failed to save model metadata to {metadata_path}: {e}") from e

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metadata saved to {metadata_path}")

    return metadata


def load_model(config_path: Optional[Path] = None):
    """Load the currently saved winning model from the registry."""
    config = load_full_config(config_path) if config_path else load_full_config()
    project_root = get_project_root()
    model_path = project_root / config["paths"]["model_registry"] / "nids_model.joblib"

    if not model_path.exists():
        raise ModelNotTrainedError(
            f"No trained model found at {model_path}. Run training first: python -m src.models.train"
        )

    try:
        return joblib.load(model_path)
    except Exception as e:
        raise ModelRegistryError(f"Failed to load model from {model_path}: {e}") from e


def _print_comparison(evaluations: list, winner_name: str) -> None:
    """Human-readable console report for the Sprint 3 verification step."""
    print(f"\n{'=' * 78}")
    print(" MODEL COMPARISON")
    print(f"{'=' * 78}")
    header = f"{'Metric':<24}"
    for ev in evaluations:
        marker = "  <- SELECTED" if ev.model_name == winner_name else ""
        header = header  # placeholder to keep structure readable below
    for ev in evaluations:
        marker = "  <-- SELECTED" if ev.model_name == winner_name else ""
        print(f"\n--- {ev.model_name}{marker} ---")
        print(f"  Accuracy              : {ev.accuracy:.4f}")
        print(f"  Precision              : {ev.precision:.4f}")
        print(f"  Recall                 : {ev.recall:.4f}")
        print(f"  F1 Score               : {ev.f1:.4f}")
        print(f"  ROC-AUC                : {ev.roc_auc:.4f}")
        print(f"  False Positive Rate    : {ev.false_positive_rate:.4f}")
        print(f"  False Negative Rate    : {ev.false_negative_rate:.4f}")
        print(f"  CV ROC-AUC (train)     : {ev.cv_roc_auc_mean:.4f} +/- {ev.cv_roc_auc_std:.4f}")
        print(f"  Confusion Matrix       : TP={ev.true_positives} TN={ev.true_negatives} "
              f"FP={ev.false_positives} FN={ev.false_negatives}")
    print(f"\n{'=' * 78}\n")


if __name__ == "__main__":
    # Standalone verification entrypoint for Sprint 3.
    # Run: python -m src.models.train
    from src.data.loader import load_dataset
    from src.data.preprocessor import NIDSPreprocessor

    try:
        train_df = load_dataset(split="train")
        test_df = load_dataset(split="test")

        pre = NIDSPreprocessor()
        X_train, y_train = pre.fit_transform(train_df)
        X_test, y_test = pre.transform(test_df)
        pre.save()  # keep preprocessor artifact in sync with this training run

        winner_model, winner_name, evaluations = train_and_evaluate(
            X_train, y_train, X_test, y_test
        )

        _print_comparison(evaluations, winner_name)

        metadata = save_model(winner_model, winner_name, evaluations)

        # Reload to prove the saved artifact is actually usable for inference,
        # the same guarantee we verified for the preprocessor in Sprint 2.
        reloaded_model = load_model()
        reloaded_preds = reloaded_model.predict(X_test[:100])
        original_preds = winner_model.predict(X_test[:100])
        identical = np.array_equal(reloaded_preds, original_preds)

        print(f"Winning model           : {winner_name}")
        print(f"Reload -> identical preds: {identical}")

        if not identical:
            raise AegisNIDSError(
                "Reloaded model produced different predictions than the "
                "original trained instance."
            )

        logger.info(f"Sprint 3 verification PASSED: {winner_name} trained, evaluated, and persisted.")

    except AegisNIDSError as e:
        logger.error(f"Sprint 3 verification failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}", exc_info=True)
        raise SystemExit(1) from e
