"""Numerical statistics analyzer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from scipy import stats

from leda.core.base_analyzer import BaseAnalyzer


class NumericalAnalyzer(BaseAnalyzer):
    """
    Comprehensive numerical data analyzer.
    
    Provides detailed statistical analysis for numeric columns:
    - Descriptive statistics
    - Distribution analysis
    - Outlier detection
    - Normality tests
    - Trend analysis
    """
    
    @property
    def analyzer_name(self) -> str:
        return "numerical_stats"
    
    def can_analyze(self, series: pd.Series) -> bool:
        """Check if series contains numeric data."""
        return pd.api.types.is_numeric_dtype(series)
    
    def _analyze_impl(self, series: pd.Series) -> Dict[str, Any]:
        """Implement comprehensive numerical analysis."""
        # Remove missing values for calculations
        clean_series = series.dropna()
        
        if clean_series.empty:
            return {"error": "No valid numeric values found"}
        
        results = {}
        
        # Descriptive statistics
        results["descriptive"] = self._descriptive_stats(clean_series)
        
        # Distribution analysis
        results["distribution"] = self._distribution_analysis(clean_series)
        
        # Outlier detection
        results["outliers"] = self._outlier_analysis(clean_series)
        
        # Quantiles and percentiles
        results["quantiles"] = self._quantile_analysis(clean_series)
        
        # Statistical tests (if applicable)
        if len(clean_series) >= 8:  # Minimum sample size for tests
            results["tests"] = self._statistical_tests(clean_series)
        
        return results
    
    def _descriptive_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive descriptive statistics."""
        return {
            "count": int(len(series)),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "mode": self._safe_mode(series),
            "std": float(series.std()),
            "var": float(series.var()),
            "min": float(series.min()),
            "max": float(series.max()),
            "range": float(series.max() - series.min()),
            "sum": float(series.sum()),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
            "coefficient_of_variation": float(series.std() / series.mean()) if series.mean() != 0 else float('inf'),
        }
    
    def _safe_mode(self, series: pd.Series) -> Optional[float]:
        """Calculate mode with error handling."""
        try:
            mode_result = series.mode()
            return float(mode_result.iloc[0]) if not mode_result.empty else None
        except Exception:
            return None
    
    def _distribution_analysis(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze distribution characteristics."""
        # Basic distribution info
        dist_info = {
            "is_symmetric": abs(series.skew()) < 0.5,
            "is_normal_like": abs(series.skew()) < 0.5 and abs(series.kurtosis()) < 3,
            "skew_interpretation": self._interpret_skewness(series.skew()),
            "kurtosis_interpretation": self._interpret_kurtosis(series.kurtosis()),
        }
        
        # Histogram data for visualization
        hist_data = self._histogram_data(series)
        dist_info.update(hist_data)
        
        return dist_info
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness value."""
        if skewness > 1:
            return "highly_right_skewed"
        elif skewness > 0.5:
            return "moderately_right_skewed"
        elif skewness > -0.5:
            return "approximately_symmetric"
        elif skewness > -1:
            return "moderately_left_skewed"
        else:
            return "highly_left_skewed"
    
    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpret kurtosis value."""
        if kurtosis > 3:
            return "leptokurtic"  # Heavy tails
        elif kurtosis < -3:
            return "platykurtic"  # Light tails
        else:
            return "mesokurtic"   # Normal-like tails
    
    def _histogram_data(self, series: pd.Series, bins: int = 30) -> Dict[str, Any]:
        """Generate histogram data for visualization."""
        try:
            counts, bin_edges = np.histogram(series, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            return {
                "histogram": {
                    "counts": counts.tolist(),
                    "bin_edges": bin_edges.tolist(),
                    "bin_centers": bin_centers.tolist(),
                    "bin_width": float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 0.0,
                }
            }
        except Exception as e:
            return {"histogram": {"error": str(e)}}
    
    def _outlier_analysis(self, series: pd.Series) -> Dict[str, Any]:
        """Detect and analyze outliers using multiple methods."""
        outliers = {}
        
        # IQR method
        iqr_outliers = self._iqr_outliers(series)
        outliers["iqr"] = iqr_outliers
        
        # Z-score method
        zscore_outliers = self._zscore_outliers(series)
        outliers["zscore"] = zscore_outliers
        
        # Modified Z-score method (more robust)
        mod_zscore_outliers = self._modified_zscore_outliers(series)
        outliers["modified_zscore"] = mod_zscore_outliers
        
        # Summary
        outliers["summary"] = self._outlier_summary(series, iqr_outliers, zscore_outliers, mod_zscore_outliers)
        
        return outliers
    
    def _iqr_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_values = series[outlier_mask]
        
        return {
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "count": int(outlier_mask.sum()),
            "percentage": float(outlier_mask.sum() / len(series) * 100),
            "values": outlier_values.tolist()[:100],  # Limit to first 100 for performance
        }
    
    def _zscore_outliers(self, series: pd.Series, threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series))
        outlier_mask = z_scores > threshold
        outlier_values = series[outlier_mask]
        
        return {
            "threshold": threshold,
            "count": int(outlier_mask.sum()),
            "percentage": float(outlier_mask.sum() / len(series) * 100),
            "values": outlier_values.tolist()[:100],
            "max_zscore": float(z_scores.max()),
        }
    
    def _modified_zscore_outliers(self, series: pd.Series, threshold: float = 3.5) -> Dict[str, Any]:
        """Detect outliers using modified Z-score (median-based)."""
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        if mad == 0:
            # Fallback to standard deviation if MAD is 0
            modified_z_scores = np.abs((series - median) / series.std())
        else:
            modified_z_scores = 0.6745 * np.abs((series - median) / mad)
        
        outlier_mask = modified_z_scores > threshold
        outlier_values = series[outlier_mask]
        
        return {
            "threshold": threshold,
            "count": int(outlier_mask.sum()),
            "percentage": float(outlier_mask.sum() / len(series) * 100),
            "values": outlier_values.tolist()[:100],
            "max_modified_zscore": float(modified_z_scores.max()),
        }
    
    def _outlier_summary(self, series: pd.Series, iqr_result: Dict, zscore_result: Dict, mod_zscore_result: Dict) -> Dict[str, Any]:
        """Summarize outlier detection results."""
        total_count = len(series)
        
        return {
            "recommended_method": self.config.analysis.outlier_method,
            "consensus_outliers": min(iqr_result["count"], zscore_result["count"], mod_zscore_result["count"]),
            "any_method_outliers": max(iqr_result["count"], zscore_result["count"], mod_zscore_result["count"]),
            "outlier_impact": {
                "min_percentage": min(iqr_result["percentage"], zscore_result["percentage"], mod_zscore_result["percentage"]),
                "max_percentage": max(iqr_result["percentage"], zscore_result["percentage"], mod_zscore_result["percentage"]),
            }
        }
    
    def _quantile_analysis(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate quantiles and percentiles."""
        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        
        quantile_values = {}
        for q in quantiles:
            quantile_values[f"p{int(q*100)}"] = float(series.quantile(q))
        
        # Interquartile range
        iqr = series.quantile(0.75) - series.quantile(0.25)
        
        return {
            "quantiles": quantile_values,
            "iqr": float(iqr),
            "quartile_coefficent_of_dispersion": float(iqr / (series.quantile(0.75) + series.quantile(0.25))) if (series.quantile(0.75) + series.quantile(0.25)) != 0 else 0.0,
        }
    
    def _statistical_tests(self, series: pd.Series) -> Dict[str, Any]:
        """Perform statistical tests on the data."""
        tests = {}
        
        # Normality tests
        if len(series) >= 8:
            tests["normality"] = self._normality_tests(series)
        
        return tests
    
    def _normality_tests(self, series: pd.Series) -> Dict[str, Any]:
        """Test for normality using multiple methods."""
        normality_results = {}
        
        # Shapiro-Wilk test (recommended for small samples)
        if len(series) <= 5000:  # Shapiro-Wilk works best with smaller samples
            try:
                shapiro_stat, shapiro_p = stats.shapiro(series)
                normality_results["shapiro_wilk"] = {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "is_normal": shapiro_p > 0.05,
                }
            except Exception as e:
                normality_results["shapiro_wilk"] = {"error": str(e)}
        
        # Kolmogorov-Smirnov test
        try:
            # Standardize the data
            standardized = (series - series.mean()) / series.std()
            ks_stat, ks_p = stats.kstest(standardized, 'norm')
            normality_results["kolmogorov_smirnov"] = {
                "statistic": float(ks_stat),
                "p_value": float(ks_p),
                "is_normal": ks_p > 0.05,
            }
        except Exception as e:
            normality_results["kolmogorov_smirnov"] = {"error": str(e)}
        
        # D'Agostino-Pearson test
        if len(series) >= 20:
            try:
                dagostino_stat, dagostino_p = stats.normaltest(series)
                normality_results["dagostino_pearson"] = {
                    "statistic": float(dagostino_stat),
                    "p_value": float(dagostino_p),
                    "is_normal": dagostino_p > 0.05,
                }
            except Exception as e:
                normality_results["dagostino_pearson"] = {"error": str(e)}
        
        return normality_results
