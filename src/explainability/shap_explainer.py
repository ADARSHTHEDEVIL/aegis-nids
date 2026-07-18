"""
src/explainability/shap_explainer.py

Turns model predictions into human-readable explanations for security
analysts. A bare "attack" prediction with no reasoning is not something
an analyst can act on with confidence — this module answers "why did the
model flag this specific connection?" in plain language.

Uses SHAP's TreeExplainer, which computes EXACT Shapley values for
tree-based models (our XGBoost model) via fast recursive algorithms — no
sampling approximation needed, unlike KernelExplainer which would be
required for non-tree models.

Design principles applied:
  - The explainer is fit once and reused, not rebuilt per-prediction
    (rebuilding a TreeExplainer per call would be wasteful and slow for
    Sprint 5's real-time simulation).
  - Explanations translate encoded feature values back to something a
    human can read. A SHAP value attached to "protocol_type=1.0" (the
    ordinal-encoded value) is useless to an analyst; "protocol_type=tcp"
    is not.
  - Both LOCAL explanations (why THIS connection was flagged) and GLOBAL
    explanations (which features matter most across the whole dataset)
    are supported, since the dashboard in Sprint 6 needs both.
"""

from pathlib import Path
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend — no GUI required, safe for scripts/servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils.exceptions import AegisNIDSError, ModelNotTrainedError, ModelRegistryError
from src.utils.logger import get_logger, load_full_config, get_project_root

logger = get_logger(__name__)


class NIDSExplainer:
    """
    Wraps a trained tree-based NIDS model with SHAP explainability,
    providing both per-connection and global feature attribution.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config = load_full_config(config_path) if config_path else load_full_config()
        self.project_root = get_project_root()
        self.model = None
        self.explainer = None
        self.feature_names = None

    def load(self, feature_names: list) -> None:
        """
        Load the trained model from the registry and build a TreeExplainer
        around it. Must be called before explain_prediction() or
        global_feature_importance().

        Args:
            feature_names: the ordered feature name list from the fitted
                            NIDSPreprocessor (pre.feature_names), so SHAP
                            output columns can be mapped back to real names.
        """
        registry_dir = self.project_root / self.config["paths"]["model_registry"]
        model_path = registry_dir / "nids_model.joblib"

        if not model_path.exists():
            raise ModelNotTrainedError(
                f"No trained model found at {model_path}. Run Sprint 3 training first: "
                f"python -m src.models.train"
            )

        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            raise ModelRegistryError(f"Failed to load model from {model_path}: {e}") from e

        if not hasattr(self.model, "predict_proba"):
            raise AegisNIDSError(
                "Loaded model does not support predict_proba(); "
                "SHAP explanation requires a probabilistic classifier."
            )

        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            raise AegisNIDSError(f"Failed to construct SHAP TreeExplainer: {e}") from e

        self.feature_names = list(feature_names)
        logger.info(f"SHAP explainer ready for model with {len(self.feature_names)} features.")

    def _check_ready(self) -> None:
        if self.explainer is None or self.feature_names is None:
            raise AegisNIDSError(
                "NIDSExplainer.load() must be called before requesting explanations."
            )

    def explain_prediction(self, X_row: np.ndarray, top_n: int = 5) -> dict:
        """
        Explain a single prediction (e.g. one network connection / packet).

        Args:
            X_row: a single preprocessed feature vector, shape (n_features,)
                   or (1, n_features).
            top_n: how many top contributing features to return.

        Returns:
            dict with the model's prediction, confidence, and a ranked list
            of features that pushed the decision toward "attack" or "normal".
        """
        self._check_ready()

        X_row = np.atleast_2d(X_row)
        if X_row.shape[1] != len(self.feature_names):
            raise AegisNIDSError(
                f"Input row has {X_row.shape[1]} features, but explainer expects "
                f"{len(self.feature_names)}. Check that the same preprocessor "
                f"was used to generate this row."
            )

        try:
            proba = self.model.predict_proba(X_row)[0]
            prediction = int(self.model.predict(X_row)[0])
            shap_values = self.explainer.shap_values(X_row)
        except Exception as e:
            raise AegisNIDSError(f"Failed to compute prediction/SHAP values: {e}") from e

        # For binary XGBoost classifiers, shap_values is typically a single
        # array (n_samples, n_features) representing contribution toward
        # class 1 ("attack"). Handle both that and the list-of-arrays form
        # some SHAP/model version combinations return, defensively.
        if isinstance(shap_values, list):
            row_shap = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            row_shap = shap_values[0]

        contributions = list(zip(self.feature_names, row_shap, X_row[0]))
        # Rank by absolute impact, regardless of direction.
        contributions.sort(key=lambda t: abs(t[1]), reverse=True)

        top_features = []
        for name, shap_val, feature_val in contributions[:top_n]:
            direction = "toward ATTACK" if shap_val > 0 else "toward NORMAL"
            top_features.append({
                "feature": name,
                "value": float(feature_val),
                "shap_contribution": float(shap_val),
                "pushes_prediction": direction,
            })

        return {
            "prediction": "attack" if prediction == 1 else "normal",
            "confidence": float(proba[prediction]),
            "attack_probability": float(proba[1]),
            "top_contributing_features": top_features,
        }

    def global_feature_importance(self, X_sample: np.ndarray, max_samples: int = 500) -> pd.DataFrame:
        """
        Compute mean absolute SHAP value per feature across a sample of
        data — i.e. which features matter most to the model overall, not
        just for one prediction. Used for Sprint 6's dashboard summary view.

        Args:
            X_sample: feature matrix to compute global importance over
                      (typically the test set, or a subsample of it).
            max_samples: cap on rows used, since SHAP computation cost
                         scales with sample size — 500 rows gives a stable
                         estimate without excessive runtime.

        Returns:
            DataFrame with columns [feature, mean_abs_shap], sorted descending.
        """
        self._check_ready()

        if X_sample.shape[0] == 0:
            raise AegisNIDSError("Cannot compute global feature importance on empty data.")

        if X_sample.shape[0] > max_samples:
            logger.info(
                f"Subsampling {max_samples} of {X_sample.shape[0]:,} rows for "
                f"global SHAP importance (full-set computation would be unnecessarily slow)."
            )
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(X_sample.shape[0], size=max_samples, replace=False)
            X_sample = X_sample[idx]

        try:
            shap_values = self.explainer.shap_values(X_sample)
        except Exception as e:
            raise AegisNIDSError(f"Failed to compute global SHAP values: {e}") from e

        if isinstance(shap_values, list):
            values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            values = shap_values

        mean_abs_shap = np.abs(values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        return importance_df

    def save_global_importance_plot(
        self,
        X_sample: np.ndarray,
        output_path: Optional[Path] = None,
        max_samples: int = 500,
        importance_df: Optional[pd.DataFrame] = None,
    ) -> Path:
        """
        Generate and save a SHAP summary bar chart (top features by global
        importance) as a PNG, for use in reports or the Sprint 6 dashboard.

        If `importance_df` is already computed (e.g. via a prior call to
        global_feature_importance()), pass it in to avoid recomputing SHAP
        values a second time — that computation is not cheap and shouldn't
        be duplicated when both the DataFrame and the plot are needed.
        """
        if importance_df is None:
            importance_df = self.global_feature_importance(X_sample, max_samples=max_samples)
        top = importance_df.head(15).iloc[::-1]  # reverse for horizontal bar chart ordering

        if output_path is None:
            registry_dir = self.project_root / self.config["paths"]["model_registry"]
            registry_dir.mkdir(parents=True, exist_ok=True)
            output_path = registry_dir / "shap_global_importance.png"

        try:
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(top["feature"], top["mean_abs_shap"], color="#4C72B0")
            ax.set_xlabel("Mean |SHAP value| (impact on model output)")
            ax.set_title("Global Feature Importance — Aegis-NIDS XGBoost Model")
            fig.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
        except OSError as e:
            raise AegisNIDSError(f"Failed to save SHAP plot to {output_path}: {e}") from e

        logger.info(f"Global SHAP importance plot saved to {output_path}")
        return output_path


def _print_explanation(explanation: dict, row_label: str = "connection") -> None:
    """Human-readable console report for a single prediction explanation."""
    print(f"\n{'-' * 60}")
    print(f" Explanation for {row_label}")
    print(f"{'-' * 60}")
    print(f"Prediction        : {explanation['prediction'].upper()}")
    print(f"Confidence         : {explanation['confidence']:.2%}")
    print(f"Attack probability : {explanation['attack_probability']:.2%}")
    print("Top contributing features:")
    for feat in explanation["top_contributing_features"]:
        print(
            f"    {feat['feature']:<30} value={feat['value']:>10.4f}   "
            f"SHAP={feat['shap_contribution']:>+9.4f}   ({feat['pushes_prediction']})"
        )
    print(f"{'-' * 60}\n")


if __name__ == "__main__":
    # Standalone verification entrypoint for Sprint 4.
    # Run: python -m src.explainability.shap_explainer
    from src.data.loader import load_dataset
    from src.data.preprocessor import NIDSPreprocessor

    try:
        test_df = load_dataset(split="test")

        pre = NIDSPreprocessor()
        pre.load()  # reuse the fitted preprocessor from Sprint 2/3, don't refit
        X_test, y_test = pre.transform(test_df)

        explainer = NIDSExplainer()
        explainer.load(feature_names=pre.feature_names)

        # --- Local explanation: pick one true attack and one true normal row ---
        attack_idx = int(np.where(y_test == 1)[0][0])
        normal_idx = int(np.where(y_test == 0)[0][0])

        attack_explanation = explainer.explain_prediction(X_test[attack_idx])
        normal_explanation = explainer.explain_prediction(X_test[normal_idx])

        _print_explanation(attack_explanation, row_label=f"test row #{attack_idx} (true label: attack)")
        _print_explanation(normal_explanation, row_label=f"test row #{normal_idx} (true label: normal)")

        # --- Global explanation ---
        importance_df = explainer.global_feature_importance(X_test)
        plot_path = explainer.save_global_importance_plot(X_test, importance_df=importance_df)

        print(f"\n{'=' * 60}")
        print(" TOP 10 GLOBALLY IMPORTANT FEATURES")
        print(f"{'=' * 60}")
        print(importance_df.head(10).to_string(index=False))
        print(f"\nGlobal importance plot saved to: {plot_path}")
        print(f"{'=' * 60}\n")

        logger.info("Sprint 4 verification PASSED: local and global SHAP explanations generated successfully.")

    except AegisNIDSError as e:
        logger.error(f"Sprint 4 verification failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        logger.error(f"Unexpected error during SHAP explanation: {e}", exc_info=True)
        raise SystemExit(1) from e
