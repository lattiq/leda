"""Basic statistics analyzer for all data types."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import numpy as np

from leda.core.base_analyzer import BaseAnalyzer


class BasicStatsAnalyzer(BaseAnalyzer):
    """
    Universal basic statistics analyzer.
    
    Provides fundamental statistics applicable to all data types:
    - Type information
    - Unique values
    - Missing data
    - Duplicates
    - Memory usage
    """
    
    @property
    def analyzer_name(self) -> str:
        return "basic_stats"
    
    def can_analyze(self, series: pd.Series) -> bool:
        """BasicStats can analyze any pandas Series."""
        return True
    
    def _analyze_impl(self, series: pd.Series) -> Dict[str, Any]:
        """Implement basic statistics analysis."""
        total_count = len(series)
        null_count = series.isnull().sum()
        valid_count = total_count - null_count
        
        # Basic counts and percentages
        basic_stats = {
            "count": {
                "total": int(total_count),
                "valid": int(valid_count),
                "missing": int(null_count),
            },
            "percentages": {
                "valid": float(valid_count / total_count * 100) if total_count > 0 else 0.0,
                "missing": float(null_count / total_count * 100) if total_count > 0 else 0.0,
            }
        }
        
        # Type information
        dtype_info = self._analyze_dtype(series)
        basic_stats.update(dtype_info)
        
        # Uniqueness analysis
        if valid_count > 0:
            uniqueness_info = self._analyze_uniqueness(series.dropna())
            basic_stats.update(uniqueness_info)
        
        # Memory usage
        memory_info = self._analyze_memory(series)
        basic_stats.update(memory_info)
        
        # Data quality indicators
        quality_info = self._analyze_quality(series)
        basic_stats.update(quality_info)
        
        return basic_stats
    
    def _analyze_dtype(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze data type information."""
        dtype_str = str(series.dtype)
        
        # Categorize dtype
        if pd.api.types.is_numeric_dtype(series):
            category = "numeric"
            subcategory = "integer" if pd.api.types.is_integer_dtype(series) else "float"
        elif pd.api.types.is_datetime64_any_dtype(series):
            category = "datetime"
            subcategory = "datetime64"
        elif pd.api.types.is_categorical_dtype(series):
            category = "categorical"
            subcategory = "category"
        elif pd.api.types.is_bool_dtype(series):
            category = "boolean"
            subcategory = "bool"
        else:
            category = "object"
            subcategory = "string" if self._is_string_like(series) else "mixed"
        
        return {
            "type_info": {
                "dtype": dtype_str,
                "category": category,
                "subcategory": subcategory,
                "pandas_dtype": str(type(series.dtype)),
            }
        }
    
    def _is_string_like(self, series: pd.Series) -> bool:
        """Check if object series contains primarily strings."""
        if series.empty:
            return False
        
        sample = series.dropna().head(100)
        if sample.empty:
            return False
        
        string_count = sum(isinstance(x, str) for x in sample)
        return string_count / len(sample) > 0.8
    
    def _analyze_uniqueness(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze uniqueness patterns."""
        if series.empty:
            return {"uniqueness": {}}
        
        unique_count = series.nunique()
        total_count = len(series)
        
        # Calculate uniqueness ratio
        uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
        
        # Determine uniqueness category
        if uniqueness_ratio == 1.0:
            uniqueness_category = "unique"
        elif uniqueness_ratio > 0.95:
            uniqueness_category = "high_cardinality"
        elif uniqueness_ratio > 0.5:
            uniqueness_category = "medium_cardinality"
        elif uniqueness_ratio > 0.1:
            uniqueness_category = "low_cardinality"
        else:
            uniqueness_category = "very_low_cardinality"
        
        # Check for potential identifiers
        is_potential_id = (
            uniqueness_ratio > 0.95 and 
            unique_count > 10 and
            self._looks_like_identifier(series)
        )
        
        return {
            "uniqueness": {
                "unique_count": int(unique_count),
                "uniqueness_ratio": float(uniqueness_ratio),
                "category": uniqueness_category,
                "is_potential_identifier": bool(is_potential_id),
            }
        }
    
    def _looks_like_identifier(self, series: pd.Series) -> bool:
        """Heuristic to detect potential identifier columns."""
        if series.empty:
            return False
        
        sample = series.dropna().head(100)
        if sample.empty:
            return False
        
        # Check for common ID patterns
        if pd.api.types.is_numeric_dtype(series):
            # Sequential numeric IDs
            if len(sample) > 10:
                diffs = sample.diff().dropna()
                return bool(diffs.abs().median() <= 1.0)
        
        if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            # String IDs with consistent patterns
            sample_str = sample.astype(str)
            lengths = sample_str.str.len()
            # Consistent length suggests structured IDs
            return bool(lengths.std() < 2.0 and lengths.mean() > 3)
        
        return False
    
    def _analyze_memory(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze memory usage."""
        memory_bytes = series.memory_usage(deep=True)
        
        return {
            "memory": {
                "bytes": int(memory_bytes),
                "bytes_per_value": float(memory_bytes / len(series)) if len(series) > 0 else 0.0,
                "human_readable": self._format_bytes(memory_bytes),
            }
        }
    
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"
    
    def _analyze_quality(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze data quality indicators."""
        total_count = len(series)
        null_count = series.isnull().sum()
        
        quality_score = 1.0
        issues = []
        
        # Missing data penalty
        missing_ratio = null_count / total_count if total_count > 0 else 0
        if missing_ratio > self.config.analysis.missing_threshold:
            quality_score -= missing_ratio * 0.5
            issues.append(f"High missing data: {missing_ratio:.1%}")
        
        # Duplicate analysis for non-identifier columns
        if total_count > 0:
            duplicate_count = total_count - series.nunique()
            duplicate_ratio = duplicate_count / total_count
            
            # Only penalize duplicates if not an identifier column
            if duplicate_ratio > 0.1 and series.nunique() / total_count < 0.95:
                quality_score -= duplicate_ratio * 0.3
                issues.append(f"High duplicate rate: {duplicate_ratio:.1%}")
        
        # Ensure quality score stays in [0, 1]
        quality_score = max(0.0, min(1.0, quality_score))
        
        return {
            "quality": {
                "score": float(quality_score),
                "issues": issues,
                "duplicate_count": int(duplicate_count) if total_count > 0 else 0,
                "duplicate_ratio": float(duplicate_ratio) if total_count > 0 else 0.0,
            }
        }
