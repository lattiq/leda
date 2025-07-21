"""Base exception for LEDA library."""

from __future__ import annotations


class LEDAException(Exception):
    """Base exception for LEDA library."""
    
    def __init__(self, message: str, data_info: dict[str, any] | None = None):
        super().__init__(message)
        self.data_info = data_info or {}
