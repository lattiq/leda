"""Visualization-related exceptions."""

from __future__ import annotations

from .base_exception import LEDAException

class VisualizationError(LEDAException):
    """Base exception for visualization-related errors."""
    pass


class RenderingError(VisualizationError):
    """Raised when plot rendering fails."""
    pass


class ThemeError(VisualizationError):
    """Raised when theme configuration is invalid."""
    pass