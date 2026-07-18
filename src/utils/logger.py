"""
Centralized logging for Aegis-NIDS.

Every module in this project calls `get_logger(__name__)` instead of using
raw `print()`. This gives us:
  - Consistent formatting across the whole codebase
  - Rotating file logs (so logs/ doesn't grow unbounded)
  - Console + file output simultaneously
  - Easy log-level control via config.yaml

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
    logger.error("Something broke", exc_info=True)
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml

from src.utils.exceptions import ConfigError

_CONFIGURED_LOGGERS = set()

# Resolve project root regardless of current working directory
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


def _load_logging_config(config_path: Path = _DEFAULT_CONFIG_PATH) -> dict:
    """
    Load logging-related settings from config.yaml.
    Falls back to safe defaults if the config file is missing or malformed,
    so a broken config never prevents logging (and therefore debugging)
    from working.
    """
    defaults = {
        "level": "INFO",
        "log_file": "logs/aegis_nids.log",
        "max_bytes": 5 * 1024 * 1024,
        "backup_count": 3,
    }

    if not config_path.exists():
        # Not fatal — logging should never be the thing that crashes the app.
        sys.stderr.write(
            f"[logger.py] WARNING: config file not found at {config_path}. "
            f"Using default logging settings.\n"
        )
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        sys.stderr.write(
            f"[logger.py] WARNING: failed to parse config.yaml ({e}). "
            f"Using default logging settings.\n"
        )
        return defaults

    logging_cfg = raw_config.get("logging", {})
    paths_cfg = raw_config.get("paths", {})

    return {
        "level": logging_cfg.get("level", defaults["level"]),
        "log_file": paths_cfg.get("log_file", defaults["log_file"]),
        "max_bytes": logging_cfg.get("max_bytes", defaults["max_bytes"]),
        "backup_count": logging_cfg.get("backup_count", defaults["backup_count"]),
    }


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.
    Safe to call repeatedly — handlers are only attached once per logger name.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    cfg = _load_logging_config()

    level_name = str(cfg["level"]).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # --- Rotating file handler ---
    try:
        log_file_path = _PROJECT_ROOT / cfg["log_file"]
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=int(cfg["max_bytes"]),
            backupCount=int(cfg["backup_count"]),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    except OSError as e:
        # If we can't write logs to disk (permissions, read-only FS, etc.),
        # degrade gracefully to console-only logging instead of crashing.
        logger.warning(
            f"Could not attach file handler ({e}). Falling back to console-only logging."
        )

    logger.propagate = False
    _CONFIGURED_LOGGERS.add(name)

    return logger


def load_full_config(config_path: Path = _DEFAULT_CONFIG_PATH) -> dict:
    """
    Load the ENTIRE config.yaml (not just logging settings).
    Used by loader.py, preprocessor.py, train.py, etc. in later sprints.
    Raises ConfigError if the file is missing or malformed, since those
    modules genuinely cannot proceed without valid config.
    """
    if not config_path.exists():
        raise ConfigError(f"Config file not found at: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse config.yaml: {e}") from e

    if not config or not isinstance(config, dict):
        raise ConfigError("config.yaml is empty or not a valid mapping.")

    return config


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT
