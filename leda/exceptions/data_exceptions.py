"""Data-related exceptions."""

from __future__ import annotations

from .base_exception import LEDAException

class DataError(LEDAException):
    """Exception raised for data processing errors."""
    pass

class DataValidationError(DataError):
    """Raised when data fails validation checks."""
    pass


class UnsupportedDataTypeError(DataError):
    """Raised when attempting to analyze unsupported data types."""
    pass


class MissingDataError(DataError):
    """Raised when required data is missing or inaccessible."""
    pass

class SerializationError(DataError):
    """Exception raised for serialization/deserialization errors."""
    pass


class CompressionError(DataError):
    """Exception raised for compression/decompression errors."""
    pass


class FormatError(DataError):
    """Exception raised for unsupported format errors."""
    pass