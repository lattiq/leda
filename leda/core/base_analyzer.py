"""Base analyzer interface and common functionality."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import pandas as pd
from pydantic import BaseModel

from leda.config.schemas import LEDAConfig


@dataclass
class AnalysisResult:
    """
    Container for analysis results with metadata.
    
    Designed for efficient serialization and composition.
    """
    analyzer_name: str
    column_name: Optional[str]
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    
    def __post_init__(self):
        """Post-initialization setup."""
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = time.time()


class BaseAnalyzer(ABC):
    """
    Abstract base class for all data analyzers.
    
    Implements the Template Method pattern for consistent analysis workflow.
    """
    
    def __init__(self, config: LEDAConfig):
        self.config = config
        self._cache: Dict[str, Any] = {}
    
    @property
    @abstractmethod
    def analyzer_name(self) -> str:
        """Return the name of this analyzer."""
        pass
    
    @abstractmethod
    def can_analyze(self, series: pd.Series) -> bool:
        """
        Check if this analyzer can process the given series.
        
        Args:
            series: pandas Series to check
            
        Returns:
            True if analyzer can process this series
        """
        pass
    
    @abstractmethod
    def _analyze_impl(self, series: pd.Series) -> Dict[str, Any]:
        """
        Core analysis implementation.
        
        Args:
            series: pandas Series to analyze
            
        Returns:
            Dictionary of analysis results
        """
        pass
    
    def analyze(self, series: pd.Series) -> AnalysisResult:
        """
        Main analysis method implementing the template pattern.
        
        Args:
            series: pandas Series to analyze
            
        Returns:
            AnalysisResult containing the analysis data
        """
        start_time = time.perf_counter()
        
        # Validation
        if not self.can_analyze(series):
            raise ValueError(f"{self.analyzer_name} cannot analyze series '{series.name}'")
        
        # Pre-processing
        processed_series = self._preprocess(series)
        
        # Core analysis
        results = self._analyze_impl(processed_series)
        
        # Post-processing
        results = self._postprocess(results, processed_series)
        
        execution_time = time.perf_counter() - start_time
        
        return AnalysisResult(
            analyzer_name=self.analyzer_name,
            column_name=series.name,
            data=results,
            metadata={
                "series_dtype": str(series.dtype),
                "series_length": len(series),
                "null_count": series.isnull().sum(),
            },
            execution_time=execution_time,
        )
    
    def _preprocess(self, series: pd.Series) -> pd.Series:
        """
        Pre-process series before analysis.
        
        Can be overridden by subclasses for specific preprocessing needs.
        """
        return series
    
    def _postprocess(self, results: Dict[str, Any], series: pd.Series) -> Dict[str, Any]:
        """
        Post-process analysis results.
        
        Can be overridden by subclasses for result transformation.
        """
        return results
    
    def clear_cache(self) -> None:
        """Clear internal cache."""
        self._cache.clear()
