"""
src/data/preprocessor.py

Transforms raw NSL-KDD DataFrames (as returned by src.data.loader.load_dataset)
into model-ready numeric arrays, and persists the FITTED transformation state
so the exact same encoding/scaling can be replayed later — critically, on
single live packets in Sprint 5's simulation engine, not just on batch
training data.

Design principles applied:
  - fit() is only ever called on training data. transform() is called on
    both train, test, AND later on live packet features. Calling transform()
    before fit() (or on a preprocessor that failed to load) raises explicitly
    rather than producing silently wrong output.
  - Unknown categorical values at transform-time (a `service` value never
    seen during training, which WILL happen with live traffic) do not crash
    the pipeline — they're mapped to a reserved "unknown" bucket rather than
    raising or silently corrupting the feature vector.
  - NaN defensiveness: even though Sprint 1 confirmed NSL-KDD has zero NaNs,
    this pipeline does not assume that stays true (future dataset swap,
    corrupted download, live traffic with malformed fields). Numeric NaNs
    are imputed with the training median; this is logged loudly, not silent.
  - The entire fitted state (encoders + scaler + feature schema) is saved as
    ONE artifact via joblib, so there is no possibility of the scaler and
    encoder drifting out of sync across separate save files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.utils.exceptions import PreprocessingError, ModelRegistryError
from src.utils.logger import get_logger, load_full_config, get_project_root

logger = get_logger(__name__)

_UNKNOWN_CATEGORY_TOKEN = "__unknown__"


@dataclass
class PreprocessingArtifacts:
    """
    Container for everything a fitted preprocessor needs to persist.
    Bundling these together (rather than saving scaler.joblib and
    encoder.joblib separately) guarantees they can never drift out of
    sync with each other or with the feature ordering.
    """
    categorical_columns: list
    numeric_columns: list
    ordinal_encoder: OrdinalEncoder
    scaler: StandardScaler
    numeric_medians: dict          # column -> median, for NaN imputation at transform time
    feature_order: list            # exact column order the model expects
    label_mapping: dict            # {"normal": 0, "attack": 1}
    is_fitted: bool = False


class NIDSPreprocessor:
    """
    Fit/transform interface for turning raw NSL-KDD rows into model-ready
    numeric feature matrices, with a persistent save/load contract so
    training-time and inference-time preprocessing are guaranteed identical.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config = load_full_config(config_path) if config_path else load_full_config()
        self.project_root = get_project_root()

        prep_cfg = self.config.get("preprocessing", {})
        dataset_cfg = self.config.get("dataset", {})

        self.categorical_columns = prep_cfg.get(
            "categorical_columns", ["protocol_type", "service", "flag"]
        )
        self.drop_columns = prep_cfg.get("drop_columns", ["difficulty"])
        self.label_column = dataset_cfg.get("label_column", "label")
        self.binary_classification = prep_cfg.get("binary_classification", True)
        self.scaling_method = prep_cfg.get("scaling_method", "standard")

        self._artifacts: Optional[PreprocessingArtifacts] = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _split_columns(self, df: pd.DataFrame) -> tuple:
        """Determine numeric vs categorical feature columns from a raw dataframe."""
        working_cols = [
            c for c in df.columns
            if c not in self.drop_columns and c != self.label_column
        ]
        categorical = [c for c in working_cols if c in self.categorical_columns]
        numeric = [c for c in working_cols if c not in categorical]
        return categorical, numeric

    def _encode_labels(self, labels: pd.Series) -> np.ndarray:
        """Collapse NSL-KDD's 22+ attack-type labels into binary normal(0)/attack(1)."""
        if not self.binary_classification:
            raise PreprocessingError(
                "Multi-class labeling is configured but not yet implemented "
                "in this preprocessor version. Set preprocessing.binary_classification "
                "to true in config.yaml, or extend _encode_labels()."
            )
        mapping = {"normal": 0}
        # Any label that isn't literally "normal" is an attack, regardless of
        # whether the specific attack subtype appeared in training data.
        encoded = labels.apply(lambda x: 0 if x == "normal" else 1).to_numpy()
        return encoded, {"normal": 0, "attack": 1}

    def _impute_numeric_nans(self, df: pd.DataFrame, numeric_columns: list, fitting: bool) -> pd.DataFrame:
        """
        Fill NaNs in numeric columns with the training-set median.
        Logs loudly if any imputation actually occurs, since NSL-KDD should
        be clean — a NaN appearing here signals something upstream changed
        (corrupted file, different dataset, malformed live packet).
        """
        df = df.copy()
        nan_counts = df[numeric_columns].isna().sum()
        total_nans = int(nan_counts.sum())

        if total_nans > 0:
            logger.warning(
                f"Found {total_nans} NaN value(s) in numeric columns during "
                f"{'fit' if fitting else 'transform'}. Imputing with "
                f"{'newly computed' if fitting else 'stored training-set'} medians. "
                f"Affected columns: {nan_counts[nan_counts > 0].to_dict()}"
            )

        if fitting:
            medians = df[numeric_columns].median(numeric_only=True).to_dict()
            self._pending_medians = medians
        else:
            if self._artifacts is None:
                raise PreprocessingError("transform() called before fit()/load().")
            medians = self._artifacts.numeric_medians

        for col in numeric_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(medians.get(col, 0.0))

        return df

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """
        Fit the preprocessor on training data and return transformed (X, y).

        Args:
            df: raw dataframe as returned by src.data.loader.load_dataset(split="train")

        Returns:
            (X, y): X is a numpy array of shape (n_rows, n_features), y is
                    a numpy array of shape (n_rows,) with binary labels.
        """
        if self.label_column not in df.columns:
            raise PreprocessingError(
                f"Expected label column '{self.label_column}' not found in dataframe. "
                f"Available columns: {list(df.columns)}"
            )

        logger.info(f"Fitting preprocessor on {df.shape[0]:,} rows.")

        categorical_cols, numeric_cols = self._split_columns(df)
        logger.info(
            f"Identified {len(categorical_cols)} categorical columns "
            f"({categorical_cols}) and {len(numeric_cols)} numeric columns."
        )

        working_df = df.drop(columns=self.drop_columns, errors="ignore").copy()

        # --- Numeric NaN handling (fit medians) ---
        working_df = self._impute_numeric_nans(working_df, numeric_cols, fitting=True)

        # --- Categorical encoding ---
        working_df[categorical_cols] = working_df[categorical_cols].astype(str)
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,   # any category unseen at transform-time maps here
            dtype=np.float64,
        )
        try:
            encoded_cats = encoder.fit_transform(working_df[categorical_cols])
        except ValueError as e:
            raise PreprocessingError(f"Categorical encoding failed during fit: {e}") from e

        # --- Numeric scaling ---
        if self.scaling_method != "standard":
            raise PreprocessingError(
                f"Unsupported scaling_method '{self.scaling_method}'. "
                f"Only 'standard' is implemented in this version."
            )
        scaler = StandardScaler()
        try:
            scaled_numeric = scaler.fit_transform(working_df[numeric_cols])
        except ValueError as e:
            raise PreprocessingError(f"Numeric scaling failed during fit: {e}") from e

        feature_order = numeric_cols + categorical_cols
        X = np.hstack([scaled_numeric, encoded_cats])

        y, label_mapping = self._encode_labels(df[self.label_column])

        self._artifacts = PreprocessingArtifacts(
            categorical_columns=categorical_cols,
            numeric_columns=numeric_cols,
            ordinal_encoder=encoder,
            scaler=scaler,
            numeric_medians=self._pending_medians,
            feature_order=feature_order,
            label_mapping=label_mapping,
            is_fitted=True,
        )

        logger.info(
            f"Preprocessor fitted successfully. Output feature matrix shape: {X.shape}. "
            f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}"
        )

        return X, y

    def transform(self, df: pd.DataFrame) -> tuple:
        """
        Apply an already-fitted preprocessor to new data (test set, or later,
        live packet features). Unknown categorical values are safely mapped
        to -1 rather than raising.

        Returns:
            (X, y): y will be None if the dataframe has no label column
                    (as will be the case for live inference in Sprint 5).
        """
        if self._artifacts is None or not self._artifacts.is_fitted:
            raise PreprocessingError(
                "transform() called on a preprocessor that hasn't been fitted "
                "or loaded. Call fit_transform() first, or load() a saved artifact."
            )

        art = self._artifacts
        working_df = df.drop(columns=self.drop_columns, errors="ignore").copy()

        missing_cols = set(art.numeric_columns + art.categorical_columns) - set(working_df.columns)
        if missing_cols:
            raise PreprocessingError(
                f"Input data is missing columns the preprocessor expects: {missing_cols}"
            )

        working_df = self._impute_numeric_nans(working_df, art.numeric_columns, fitting=False)

        working_df[art.categorical_columns] = working_df[art.categorical_columns].astype(str)

        try:
            scaled_numeric = art.scaler.transform(working_df[art.numeric_columns])
            encoded_cats = art.ordinal_encoder.transform(working_df[art.categorical_columns])
        except ValueError as e:
            raise PreprocessingError(f"transform() failed: {e}") from e

        X = np.hstack([scaled_numeric, encoded_cats])

        y = None
        if self.label_column in df.columns:
            y, _ = self._encode_labels(df[self.label_column])

        return X, y

    def save(self, filename: str = "preprocessor.joblib") -> Path:
        """Persist the fitted preprocessor artifacts to the model registry."""
        if self._artifacts is None or not self._artifacts.is_fitted:
            raise PreprocessingError("Cannot save an unfitted preprocessor. Call fit_transform() first.")

        registry_dir = self.project_root / self.config["paths"]["model_registry"]
        registry_dir.mkdir(parents=True, exist_ok=True)
        out_path = registry_dir / filename

        try:
            joblib.dump(self._artifacts, out_path)
        except (OSError, TypeError) as e:
            raise ModelRegistryError(f"Failed to save preprocessor to {out_path}: {e}") from e

        logger.info(f"Preprocessor artifacts saved to {out_path}")
        return out_path

    def load(self, filename: str = "preprocessor.joblib") -> None:
        """Load previously fitted preprocessor artifacts from the model registry."""
        registry_dir = self.project_root / self.config["paths"]["model_registry"]
        in_path = registry_dir / filename

        if not in_path.exists():
            raise ModelRegistryError(f"No saved preprocessor found at {in_path}")

        try:
            artifacts = joblib.load(in_path)
        except Exception as e:
            raise ModelRegistryError(f"Failed to load preprocessor from {in_path}: {e}") from e

        if not isinstance(artifacts, PreprocessingArtifacts):
            raise ModelRegistryError(
                f"File at {in_path} does not contain valid PreprocessingArtifacts."
            )

        self._artifacts = artifacts
        logger.info(f"Preprocessor artifacts loaded from {in_path}")

    @property
    def feature_names(self) -> list:
        """Return the exact ordered list of feature names the model expects."""
        if self._artifacts is None:
            raise PreprocessingError("Preprocessor not fitted/loaded yet.")
        return self._artifacts.feature_order


if __name__ == "__main__":
    # Standalone verification entrypoint for Sprint 2.
    # Run: python -m src.data.preprocessor
    from src.data.loader import load_dataset
    from src.utils.exceptions import AegisNIDSError

    try:
        train_df = load_dataset(split="train")
        test_df = load_dataset(split="test")

        pre = NIDSPreprocessor()
        X_train, y_train = pre.fit_transform(train_df)
        X_test, y_test = pre.transform(test_df)

        saved_path = pre.save()

        # Reload into a fresh instance to prove the persisted artifact
        # actually reproduces identical transforms — this is the exact
        # guarantee Sprint 5's live simulation will depend on.
        pre_reloaded = NIDSPreprocessor()
        pre_reloaded.load()
        X_test_reloaded, _ = pre_reloaded.transform(test_df)

        identical = np.allclose(X_test, X_test_reloaded)

        print(f"\n{'=' * 60}")
        print(" PREPROCESSING PIPELINE VERIFICATION")
        print(f"{'=' * 60}")
        print(f"Train feature matrix shape : {X_train.shape}")
        print(f"Test feature matrix shape  : {X_test.shape}")
        print(f"Feature count              : {len(pre.feature_names)}")
        print(f"Saved artifact path        : {saved_path}")
        print(f"Reload -> identical output : {identical}")
        print(f"Train label balance        : "
              f"{dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Test label balance         : "
              f"{dict(zip(*np.unique(y_test, return_counts=True)))}")
        print(f"{'=' * 60}\n")

        if not identical:
            raise PreprocessingError(
                "Reloaded preprocessor produced different output than the "
                "original fitted instance. This would break inference "
                "consistency and must be fixed before proceeding."
            )

        logger.info("Sprint 2 verification PASSED: preprocessing pipeline is consistent and persistent.")

    except AegisNIDSError as e:
        logger.error(f"Sprint 2 verification failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing verification: {e}", exc_info=True)
        raise SystemExit(1) from e
