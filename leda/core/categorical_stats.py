"""Categorical statistics analyzer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from leda.core.base_analyzer import BaseAnalyzer


class CategoricalAnalyzer(BaseAnalyzer):
    """
    Comprehensive categorical data analyzer.
    
    Provides detailed analysis for categorical/string columns:
    - Frequency analysis
    - Cardinality assessment
    - Pattern detection
    - Text statistics
    """
    
    @property
    def analyzer_name(self) -> str:
        return "categorical_stats"
    
    def can_analyze(self, series: pd.Series) -> bool:
        """Check if series contains categorical data."""
        return (
            pd.api.types.is_categorical_dtype(series) or
            pd.api.types.is_object_dtype(series) or
            pd.api.types.is_string_dtype(series) or
            (pd.api.types.is_bool_dtype(series))
        )
    
    def _analyze_impl(self, series: pd.Series) -> Dict[str, Any]:
        """Implement comprehensive categorical analysis."""
        # Remove missing values for most calculations
        clean_series = series.dropna()
        
        if clean_series.empty:
            return {"error": "No valid categorical values found"}
        
        results = {}
        
        # Frequency analysis
        results["frequency"] = self._frequency_analysis(clean_series)
        
        # Cardinality analysis
        results["cardinality"] = self._cardinality_analysis(clean_series)
        
        # String analysis (if applicable)
        if self._is_string_data(clean_series):
            results["text"] = self._text_analysis(clean_series)
        
        # Pattern analysis
        results["patterns"] = self._pattern_analysis(clean_series)
        
        return results
    
    def _is_string_data(self, series: pd.Series) -> bool:
        """Check if series contains string data."""
        if series.empty:
            return False
        
        sample = series.head(100)
        string_count = sum(isinstance(x, str) for x in sample)
        return string_count / len(sample) > 0.8
    
    def _frequency_analysis(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze value frequencies and distributions."""
        value_counts = series.value_counts()
        total_count = len(series)
        
        # Top categories (limited by config)
        max_categories = self.config.visualization.max_categories
        top_categories = value_counts.head(max_categories)
        
        # Frequency statistics
        frequencies = value_counts.values
        frequency_stats = {
            "most_frequent": {
                "value": str(value_counts.index[0]),
                "count": int(value_counts.iloc[0]),
                "percentage": float(value_counts.iloc[0] / total_count * 100),
            },
            "least_frequent": {
                "value": str(value_counts.index[-1]),
                "count": int(value_counts.iloc[-1]),
                "percentage": float(value_counts.iloc[-1] / total_count * 100),
            },
            "frequency_distribution": {
                "mean_frequency": float(np.mean(frequencies)),
                "median_frequency": float(np.median(frequencies)),
                "std_frequency": float(np.std(frequencies)),
                "min_frequency": int(np.min(frequencies)),
                "max_frequency": int(np.max(frequencies)),
            }
        }
        
        # Value counts for top categories
        top_categories_data = []
        for value, count in top_categories.items():
            top_categories_data.append({
                "value": str(value),
                "count": int(count),
                "percentage": float(count / total_count * 100),
            })
        
        return {
            "statistics": frequency_stats,
            "top_categories": top_categories_data,
            "unique_count": int(series.nunique()),
            "total_count": int(total_count),
        }
    
    def _cardinality_analysis(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze cardinality characteristics."""
        unique_count = series.nunique()
        total_count = len(series)
        cardinality_ratio = unique_count / total_count if total_count > 0 else 0
        
        # Classify cardinality
        if cardinality_ratio >= 0.95:
            cardinality_type = "high"
            analysis = "Almost unique values - potential identifier"
        elif cardinality_ratio >= 0.5:
            cardinality_type = "medium"
            analysis = "High diversity - good for grouping analysis"
        elif cardinality_ratio >= 0.1:
            cardinality_type = "low"
            analysis = "Limited categories - suitable for categorical analysis"
        else:
            cardinality_type = "very_low"
            analysis = "Very few unique values - highly repetitive data"
        
        # Gini coefficient for inequality measurement
        value_counts = series.value_counts()
        gini_coefficient = self._calculate_gini_coefficient(value_counts.values)
        
        return {
            "unique_count": int(unique_count),
            "total_count": int(total_count),
            "cardinality_ratio": float(cardinality_ratio),
            "cardinality_type": cardinality_type,
            "analysis": analysis,
            "gini_coefficient": float(gini_coefficient),
            "concentration": {
                "top_1_percentage": float(value_counts.iloc[0] / total_count * 100) if len(value_counts) > 0 else 0.0,
                "top_5_percentage": float(value_counts.head(5).sum() / total_count * 100) if len(value_counts) > 0 else 0.0,
                "top_10_percentage": float(value_counts.head(10).sum() / total_count * 100) if len(value_counts) > 0 else 0.0,
            }
        }
    
    def _calculate_gini_coefficient(self, frequencies: np.ndarray) -> float:
        """Calculate Gini coefficient for frequency distribution."""
        if len(frequencies) <= 1:
            return 0.0
        
        # Sort frequencies
        sorted_freq = np.sort(frequencies)
        n = len(sorted_freq)
        cumsum = np.cumsum(sorted_freq)
        
        # Gini coefficient formula
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_freq))) / (n * cumsum[-1]) - (n + 1) / n
        return max(0.0, min(1.0, gini))  # Ensure [0, 1] range
    
    def _text_analysis(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze text characteristics for string data."""
        # Convert to string and get lengths
        str_series = series.astype(str)
        lengths = str_series.str.len()
        
        # Character analysis
        char_stats = {
            "length_statistics": {
                "mean": float(lengths.mean()),
                "median": float(lengths.median()),
                "std": float(lengths.std()),
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "mode": int(lengths.mode().iloc[0]) if not lengths.mode().empty else 0,
            }
        }
        
        # Common patterns
        patterns = self._analyze_text_patterns(str_series)
        char_stats["patterns"] = patterns
        
        # Character composition
        composition = self._analyze_character_composition(str_series)
        char_stats["composition"] = composition
        
        return char_stats
    
    def _analyze_text_patterns(self, str_series: pd.Series) -> Dict[str, Any]:
        """Analyze common text patterns."""
        sample_size = min(1000, len(str_series))  # Limit sample for performance
        sample = str_series.head(sample_size)
        
        patterns = {
            "email_like": int(sample.str.contains(r'@.*\.', regex=True, na=False).sum()),
            "url_like": int(sample.str.contains(r'https?://', regex=True, na=False).sum()),
            "phone_like": int(sample.str.contains(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', regex=True, na=False).sum()),
            "date_like": int(sample.str.contains(r'\d{4}[-/]\d{2}[-/]\d{2}', regex=True, na=False).sum()),
            "numeric_only": int(sample.str.match(r'^\d+', na=False).sum()),
            "alpha_only": int(sample.str.match(r'^[a-zA-Z]+', na=False).sum()),
            "alphanumeric": int(sample.str.match(r'^[a-zA-Z0-9]+', na=False).sum()),
            "contains_spaces": int(sample.str.contains(r'\s', regex=True, na=False).sum()),
            "uppercase": int(sample.str.isupper().sum()),
            "lowercase": int(sample.str.islower().sum()),
            "titlecase": int(sample.str.istitle().sum()),
        }
        
        # Convert counts to percentages
        for key, count in patterns.items():
            patterns[f"{key}_percentage"] = float(count / sample_size * 100)
        
        return patterns
    
    def _analyze_character_composition(self, str_series: pd.Series) -> Dict[str, Any]:
        """Analyze character composition of text data."""
        sample_size = min(1000, len(str_series))
        sample = str_series.head(sample_size)
        
        # Combine all text for character analysis
        all_text = ' '.join(sample.astype(str))
        
        if not all_text:
            return {}
        
        total_chars = len(all_text)
        
        composition = {
            "total_characters": total_chars,
            "unique_characters": len(set(all_text)),
            "alphabetic": sum(c.isalpha() for c in all_text),
            "numeric": sum(c.isdigit() for c in all_text),
            "whitespace": sum(c.isspace() for c in all_text),
            "punctuation": sum(not c.isalnum() and not c.isspace() for c in all_text),
        }
        
        # Convert to percentages
        for key in ["alphabetic", "numeric", "whitespace", "punctuation"]:
            composition[f"{key}_percentage"] = float(composition[key] / total_chars * 100) if total_chars > 0 else 0.0
        
        return composition
    
    def _pattern_analysis(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze patterns in categorical data."""
        patterns = {}
        
        # Repetition patterns
        value_counts = series.value_counts()
        
        # Singleton analysis (values that appear only once)
        singletons = (value_counts == 1).sum()
        patterns["singletons"] = {
            "count": int(singletons),
            "percentage": float(singletons / len(value_counts) * 100) if len(value_counts) > 0 else 0.0,
        }
        
        # Frequency distribution analysis
        freq_distribution = value_counts.value_counts().sort_index()
        patterns["frequency_distribution"] = {
            "frequencies": freq_distribution.index.tolist(),
            "counts": freq_distribution.values.tolist(),
        }
        
        # Most common frequency
        if not freq_distribution.empty:
            most_common_freq = freq_distribution.idxmax()
            patterns["most_common_frequency"] = {
                "frequency": int(most_common_freq),
                "count_of_values_with_this_frequency": int(freq_distribution[most_common_freq]),
            }
        
        # Diversity metrics
        patterns["diversity"] = {
            "simpson_diversity": self._simpson_diversity_index(value_counts),
            "shannon_entropy": self._shannon_entropy(value_counts),
            "evenness": self._calculate_evenness(value_counts),
        }
        
        return patterns
    
    def _simpson_diversity_index(self, value_counts: pd.Series) -> float:
        """Calculate Simpson's Diversity Index."""
        if value_counts.empty:
            return 0.0
        
        total = value_counts.sum()
        proportions = value_counts / total
        simpson_index = (proportions ** 2).sum()
        return float(1 - simpson_index)  # 1 - Simpson's index for diversity
    
    def _shannon_entropy(self, value_counts: pd.Series) -> float:
        """Calculate Shannon entropy."""
        if value_counts.empty:
            return 0.0
        
        total = value_counts.sum()
        proportions = value_counts / total
        # Avoid log(0) by filtering out zero proportions
        non_zero_props = proportions[proportions > 0]
        entropy = -(non_zero_props * np.log2(non_zero_props)).sum()
        return float(entropy)
    
    def _calculate_evenness(self, value_counts: pd.Series) -> float:
        """Calculate evenness (how evenly distributed the categories are)."""
        if value_counts.empty or len(value_counts) == 1:
            return 1.0
        
        shannon_entropy = self._shannon_entropy(value_counts)
        max_entropy = np.log2(len(value_counts))
        return float(shannon_entropy / max_entropy) if max_entropy > 0 else 0.0