"""
Custom exception hierarchy for Aegis-NIDS.

Using specific exception types (rather than bare Exception) lets calling
code distinguish between "file missing", "schema invalid", "model not
trained yet", etc., and react accordingly instead of swallowing every
error the same way.
"""


class AegisNIDSError(Exception):
    """Base exception for all Aegis-NIDS custom errors."""
    pass


class DatasetNotFoundError(AegisNIDSError):
    """Raised when a required dataset file cannot be located on disk."""
    pass


class SchemaValidationError(AegisNIDSError):
    """Raised when a loaded dataset does not match the expected schema
    (wrong column count, missing required columns, etc.)."""
    pass


class DataQualityError(AegisNIDSError):
    """Raised when data fails quality checks (excessive NaNs, empty
    dataframe after cleaning, all-constant columns, etc.)."""
    pass


class PreprocessingError(AegisNIDSError):
    """Raised when the preprocessing pipeline fails or is used in an
    invalid state (e.g. transform() called before fit())."""
    pass


class ModelNotTrainedError(AegisNIDSError):
    """Raised when inference is attempted on a model that hasn't been
    trained or loaded yet."""
    pass


class ModelRegistryError(AegisNIDSError):
    """Raised when saving/loading a model artifact from the registry fails."""
    pass


class PacketProcessingError(AegisNIDSError):
    """Raised when a captured/replayed packet cannot be parsed or
    converted into a feature vector."""
    pass


class ConfigError(AegisNIDSError):
    """Raised when config.yaml is missing, malformed, or missing required keys."""
    pass
