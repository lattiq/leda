"""
LattIQ EDA (LEDA) - High-performance exploratory data analysis library.

LEDA provides comprehensive data profiling, visualization, and reporting
capabilities with a focus on performance, extensibility, and cloud-native deployment.
"""

from __future__ import annotations

from ._version import __version__
from .core.data_profiler import DataProfiler
from .config.defaults import get_default_config

# Public API - Progressive disclosure principle
__all__ = [
    "__version__",
    "DataProfiler",
    "profile_data",
    "get_default_config",
]


def profile_data(
    data,
    *,
    config=None,
    output_format: str = "msgpack",
    include_visualizations: bool = True,
    **kwargs,
):
    """
    Quick entry point for data profiling - 80% use case.
    
    Args:
        data: pandas.DataFrame, polars.DataFrame, or file path
        config: Optional configuration override
        output_format: "msgpack" (default) or "json"
        include_visualizations: Generate visualization data
        **kwargs: Additional profiling options
        
    Returns:
        Analysis results in specified format
    """
    profiler = DataProfiler(config=config, **kwargs)
    return profiler.profile(
        data, 
        output_format=output_format,
        include_visualizations=include_visualizations
    )
