"""Custom exception hierarchy for academic risk data processing."""


class DataProcessingError(Exception):
    """Base exception for all data processing failures in the dashboard."""


class DataLoadError(DataProcessingError):
    """Raised when loading data from file or database fails."""


class DataValidationError(DataProcessingError):
    """Raised when data fails structural or schema validation checks."""


class DataTransformationError(DataProcessingError):
    """Raised when transforming, casting, or cleaning data fails."""


class DataSaveError(DataProcessingError):
    """Raised when writing or persisting data fails."""
