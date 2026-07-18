"""
src/data/loader.py

Responsible for ONE thing: getting a clean, schema-validated NSL-KDD
DataFrame into memory, safely.

NSL-KDD ships as headerless CSV-like text files (KDDTrain+.txt / KDDTest+.txt).
There is no bundled "names" file we can depend on downloading reliably, so
the 43-column schema (41 features + label + difficulty) is defined
explicitly here, matching the canonical NSL-KDD field definitions.

Design principles applied:
  - Never assume the file exists -> check, and offer to auto-download.
  - Never assume the download succeeds -> retry with backoff, verify size.
  - Never assume the file parses cleanly -> catch parser errors explicitly.
  - Never assume the schema is correct -> validate column count before
    proceeding, fail loudly and specifically if it isn't.
  - Never assume the data is clean -> report NaNs / dtypes rather than
    silently trusting them (actual cleaning happens in Sprint 2's
    preprocessor.py; this module only loads + reports).
"""

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.exceptions import DatasetNotFoundError, SchemaValidationError
from src.utils.logger import get_logger, load_full_config, get_project_root

logger = get_logger(__name__)

# --- Canonical NSL-KDD schema (41 features + label + difficulty = 43 cols) ---
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

EXPECTED_COLUMN_COUNT = len(NSL_KDD_COLUMNS)  # 43

# Known-good mirrors (raw.githubusercontent.com is on the allowlist).
# Multiple mirrors are tried in order for resilience against a single
# repo going away or rate-limiting us.
_MIRROR_BASE_URLS = [
    "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master",
    "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/master/NSL_KDD_Dataset",
]

_DOWNLOAD_TIMEOUT_SECONDS = 30
_DOWNLOAD_MAX_RETRIES = 3
_DOWNLOAD_RETRY_BACKOFF_SECONDS = 2


def _download_with_retries(url: str, destination: Path) -> bool:
    """
    Attempt to download a single URL to `destination` with retries.
    Returns True on success, False on failure (never raises — caller
    decides whether to try the next mirror or give up).
    """
    for attempt in range(1, _DOWNLOAD_MAX_RETRIES + 1):
        try:
            logger.info(f"Download attempt {attempt}/{_DOWNLOAD_MAX_RETRIES}: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Aegis-NIDS/1.0"})
            with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
                if response.status != 200:
                    logger.warning(f"Non-200 status ({response.status}) from {url}")
                    continue
                data = response.read()

            if len(data) < 1024:
                # Suspiciously small — likely an HTML error page, not the dataset.
                logger.warning(
                    f"Downloaded content from {url} is only {len(data)} bytes; "
                    f"likely not a valid dataset file. Skipping."
                )
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            logger.info(f"Successfully downloaded {len(data):,} bytes to {destination}")
            return True

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            logger.warning(f"Download attempt {attempt} failed for {url}: {e}")
            if attempt < _DOWNLOAD_MAX_RETRIES:
                time.sleep(_DOWNLOAD_RETRY_BACKOFF_SECONDS * attempt)
        except OSError as e:
            logger.error(f"Local filesystem error while saving download: {e}")
            return False

    return False


def _attempt_auto_download(filename: str, destination: Path) -> bool:
    """Try every known mirror in order until one succeeds."""
    logger.info(f"'{filename}' not found locally. Attempting auto-download from mirrors...")
    for base_url in _MIRROR_BASE_URLS:
        url = f"{base_url}/{filename}"
        if _download_with_retries(url, destination):
            return True
    return False


def _validate_schema(df: pd.DataFrame, source_name: str) -> None:
    """
    Raise SchemaValidationError if the loaded dataframe doesn't match the
    expected NSL-KDD structure. Catching this early prevents confusing
    downstream errors in preprocessing/training.
    """
    if df.shape[1] != EXPECTED_COLUMN_COUNT:
        raise SchemaValidationError(
            f"Schema mismatch in '{source_name}': expected "
            f"{EXPECTED_COLUMN_COUNT} columns, got {df.shape[1]}. "
            f"The file may be corrupted, truncated, or not NSL-KDD data."
        )

    if df.empty:
        raise SchemaValidationError(
            f"'{source_name}' loaded successfully but contains zero rows."
        )

    required_cols = {"label", "protocol_type", "service", "flag"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"'{source_name}' is missing required columns after assignment: {missing}"
        )


def load_dataset(
    split: str = "train",
    config_path: Optional[Path] = None,
    auto_download: bool = True,
) -> pd.DataFrame:
    """
    Load the NSL-KDD train or test split as a schema-validated DataFrame.

    Args:
        split: "train" or "test".
        config_path: optional override for config.yaml location (mainly for tests).
        auto_download: if True, attempts to download the file from known
                        mirrors when it isn't found locally.

    Returns:
        pd.DataFrame with 43 named columns (41 features + label + difficulty).

    Raises:
        DatasetNotFoundError: file missing locally and download failed/disabled.
        SchemaValidationError: file loaded but doesn't match expected NSL-KDD schema.
        ValueError: invalid `split` argument.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got: {split!r}")

    config = load_full_config(config_path) if config_path else load_full_config()
    project_root = get_project_root()

    raw_dir = project_root / config["paths"]["raw_dir"]
    filename = config["dataset"]["train_file"] if split == "train" else config["dataset"]["test_file"]
    file_path = raw_dir / filename

    if not file_path.exists():
        if not auto_download:
            raise DatasetNotFoundError(
                f"Dataset file not found at {file_path} and auto_download=False."
            )
        success = _attempt_auto_download(filename, file_path)
        if not success:
            raise DatasetNotFoundError(
                f"Could not locate '{filename}' locally and all download mirrors "
                f"failed. Please manually place the NSL-KDD files in "
                f"{raw_dir}/ (expected: KDDTrain+.txt and KDDTest+.txt)."
            )

    logger.info(f"Loading '{split}' split from {file_path}")

    try:
        df = pd.read_csv(
            file_path,
            header=None,
            names=NSL_KDD_COLUMNS,
            skipinitialspace=True,
            na_values=["", " ", "?", "NA", "N/A", "null", "NULL"],
        )
    except pd.errors.ParserError as e:
        raise SchemaValidationError(
            f"Failed to parse '{file_path}' as CSV: {e}"
        ) from e
    except UnicodeDecodeError as e:
        logger.warning(f"UTF-8 decode failed for {file_path}, retrying with latin-1: {e}")
        try:
            df = pd.read_csv(
                file_path,
                header=None,
                names=NSL_KDD_COLUMNS,
                skipinitialspace=True,
                na_values=["", " ", "?", "NA", "N/A", "null", "NULL"],
                encoding="latin-1",
            )
        except Exception as e2:
            raise SchemaValidationError(
                f"Failed to parse '{file_path}' with both utf-8 and latin-1 encodings: {e2}"
            ) from e2

    _validate_schema(df, source_name=str(file_path))

    # NSL-KDD labels include specific attack names (e.g. 'neptune', 'smurf').
    # Strip any trailing '.' artifacts some mirrors include (a legacy quirk
    # inherited from the original KDD Cup 99 format).
    df["label"] = df["label"].astype(str).str.strip().str.rstrip(".")

    logger.info(
        f"Loaded '{split}' split successfully: {df.shape[0]:,} rows, {df.shape[1]} columns."
    )

    return df


def summarize_dataset(df: pd.DataFrame, split_name: str = "dataset") -> dict:
    """
    Produce a diagnostic summary of a loaded dataset: shape, class balance,
    missing values, dtypes. Used for the Sprint 1 verification step and
    useful again in Sprint 2 before/after cleaning.
    """
    summary = {
        "split_name": split_name,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "total_nan_count": int(df.isna().sum().sum()),
        "columns_with_nans": {
            col: int(count) for col, count in df.isna().sum().items() if count > 0
        },
        "binary_class_distribution": {
            "normal": int((df["label"] == "normal").sum()),
            "attack": int((df["label"] != "normal").sum()),
        },
        "n_unique_attack_types": int(df.loc[df["label"] != "normal", "label"].nunique()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    return summary


def _print_summary(summary: dict) -> None:
    """Human-readable console report for the verification step."""
    print(f"\n{'=' * 60}")
    print(f" DATASET SUMMARY: {summary['split_name']}")
    print(f"{'=' * 60}")
    print(f"Rows                : {summary['n_rows']:,}")
    print(f"Columns             : {summary['n_columns']}")
    print(f"Duplicate rows      : {summary['n_duplicate_rows']:,}")
    print(f"Total NaN cells     : {summary['total_nan_count']:,}")
    if summary["columns_with_nans"]:
        print("Columns with NaNs   :")
        for col, count in summary["columns_with_nans"].items():
            print(f"    - {col}: {count}")
    else:
        print("Columns with NaNs   : none")
    print(f"Class distribution  : "
          f"normal={summary['binary_class_distribution']['normal']:,}  "
          f"attack={summary['binary_class_distribution']['attack']:,}")
    print(f"Unique attack types : {summary['n_unique_attack_types']}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # Standalone verification entrypoint for Sprint 1.
    # Run: python -m src.data.loader
    try:
        train_df = load_dataset(split="train")
        test_df = load_dataset(split="test")

        train_summary = summarize_dataset(train_df, split_name="KDDTrain+")
        test_summary = summarize_dataset(test_df, split_name="KDDTest+")

        _print_summary(train_summary)
        _print_summary(test_summary)

        logger.info("Sprint 1 verification PASSED: both splits loaded and validated cleanly.")

    except DatasetNotFoundError as e:
        logger.error(f"Dataset acquisition failed: {e}")
        raise SystemExit(1) from e
    except SchemaValidationError as e:
        logger.error(f"Dataset schema validation failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        logger.error(f"Unexpected error during dataset loading: {e}", exc_info=True)
        raise SystemExit(1) from e
