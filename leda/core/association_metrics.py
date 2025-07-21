"""
Association metrics analyzer for cross-variable relationships.

This module provides comprehensive analysis of relationships between variables
including correlations, categorical associations, and mixed-type relationships.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr, kendalltau

from ..config.schemas import LEDAConfig
from ..exceptions.data_exceptions import InsufficientDataError, InvalidDataTypeError
from .base_analyzer import BaseAnalyzer, AnalysisResult


class AssociationType(Enum):
    """Types of variable associations."""
    NUMERICAL_NUMERICAL = "numerical_numerical"
    CATEGORICAL_CATEGORICAL = "categorical_categorical"
    NUMERICAL_CATEGORICAL = "numerical_categorical"
    MIXED = "mixed"


class CorrelationMethod(Enum):
    """Correlation calculation methods."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


@dataclass
class AssociationResult:
    """Result of an association analysis between two variables."""
    var1: str
    var2: str
    association_type: AssociationType
    method: str
    statistic: float
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size: int = 0
    interpretation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "var1": self.var1,
            "var2": self.var2,
            "association_type": self.association_type.value,
            "method": self.method,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "confidence_interval": list(self.confidence_interval) if self.confidence_interval else None,
            "sample_size": self.sample_size,
            "interpretation": self.interpretation
        }


class BaseAssociationAnalyzer(ABC):
    """Abstract base class for association analyzers."""
    
    def __init__(self, config: LEDAConfig):
        self.config = config
        self.min_sample_size = config.analysis.min_sample_size
        self.significance_level = config.analysis.significance_level
    
    @abstractmethod
    def can_analyze(self, var1: pd.Series, var2: pd.Series) -> bool:
        """Check if this analyzer can handle the given variable types."""
        pass
    
    @abstractmethod
    def analyze(self, var1: pd.Series, var2: pd.Series) -> AssociationResult:
        """Perform association analysis between two variables."""
        pass
    
    def _validate_data(self, var1: pd.Series, var2: pd.Series) -> None:
        """Validate input data for analysis."""
        if len(var1) != len(var2):
            raise InvalidDataTypeError("Variables must have the same length")
        
        # Remove missing values
        mask = ~(var1.isna() | var2.isna())
        valid_count = mask.sum()
        
        if valid_count < self.min_sample_size:
            raise InsufficientDataError(
                f"Insufficient valid observations: {valid_count} < {self.min_sample_size}"
            )
    
    def _interpret_correlation(self, correlation: float, method: str = "pearson") -> str:
        """Provide interpretation for correlation coefficient."""
        abs_corr = abs(correlation)
        
        if abs_corr < 0.1:
            strength = "negligible"
        elif abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.5:
            strength = "moderate"
        elif abs_corr < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if correlation > 0 else "negative"
        
        return f"{strength.capitalize()} {direction} {method} correlation"


class NumericalAssociationAnalyzer(BaseAssociationAnalyzer):
    """Analyzer for numerical-numerical variable associations."""
    
    def can_analyze(self, var1: pd.Series, var2: pd.Series) -> bool:
        """Check if both variables are numerical."""
        return (
            pd.api.types.is_numeric_dtype(var1) and 
            pd.api.types.is_numeric_dtype(var2)
        )
    
    def analyze(self, var1: pd.Series, var2: pd.Series) -> AssociationResult:
        """Analyze correlation between numerical variables."""
        self._validate_data(var1, var2)
        
        # Remove missing values
        mask = ~(var1.isna() | var2.isna())
        clean_var1 = var1[mask]
        clean_var2 = var2[mask]
        
        # Choose correlation method based on data distribution
        method = self._select_correlation_method(clean_var1, clean_var2)
        
        if method == CorrelationMethod.PEARSON:
            correlation, p_value = pearsonr(clean_var1, clean_var2)
            method_name = "pearson"
        elif method == CorrelationMethod.SPEARMAN:
            correlation, p_value = spearmanr(clean_var1, clean_var2)
            method_name = "spearman"
        else:  # Kendall
            correlation, p_value = kendalltau(clean_var1, clean_var2)
            method_name = "kendall"
        
        # Calculate confidence interval for Pearson correlation
        confidence_interval = None
        if method == CorrelationMethod.PEARSON and len(clean_var1) > 3:
            confidence_interval = self._correlation_confidence_interval(
                correlation, len(clean_var1)
            )
        
        return AssociationResult(
            var1=var1.name,
            var2=var2.name,
            association_type=AssociationType.NUMERICAL_NUMERICAL,
            method=method_name,
            statistic=correlation,
            p_value=p_value,
            effect_size=correlation**2,  # R-squared as effect size
            confidence_interval=confidence_interval,
            sample_size=len(clean_var1),
            interpretation=self._interpret_correlation(correlation, method_name)
        )
    
    def _select_correlation_method(self, var1: pd.Series, var2: pd.Series) -> CorrelationMethod:
        """Select appropriate correlation method based on data characteristics."""
        # Check for normality (simplified test)
        if len(var1) > 20:  # Need sufficient sample size for normality test
            try:
                _, p1 = stats.shapiro(var1.sample(min(5000, len(var1))))
                _, p2 = stats.shapiro(var2.sample(min(5000, len(var2))))
                
                # If both appear normal, use Pearson
                if p1 > 0.05 and p2 > 0.05:
                    return CorrelationMethod.PEARSON
            except:
                pass  # Fall back to non-parametric
        
        # Default to Spearman for robustness
        return CorrelationMethod.SPEARMAN
    
    def _correlation_confidence_interval(
        self, correlation: float, n: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for Pearson correlation."""
        # Fisher z-transformation
        z = np.arctanh(correlation)
        se = 1 / np.sqrt(n - 3)
        
        # Critical value
        alpha = 1 - confidence
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        # Transform back
        lower_z = z - z_critical * se
        upper_z = z + z_critical * se
        
        return (np.tanh(lower_z), np.tanh(upper_z))


class CategoricalAssociationAnalyzer(BaseAssociationAnalyzer):
    """Analyzer for categorical-categorical variable associations."""
    
    def can_analyze(self, var1: pd.Series, var2: pd.Series) -> bool:
        """Check if both variables are categorical."""
        return (
            not pd.api.types.is_numeric_dtype(var1) and 
            not pd.api.types.is_numeric_dtype(var2)
        )
    
    def analyze(self, var1: pd.Series, var2: pd.Series) -> AssociationResult:
        """Analyze association between categorical variables."""
        self._validate_data(var1, var2)
        
        # Remove missing values
        mask = ~(var1.isna() | var2.isna())
        clean_var1 = var1[mask]
        clean_var2 = var2[mask]
        
        # Create contingency table
        contingency_table = pd.crosstab(clean_var1, clean_var2)
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Cramér's V (effect size)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        
        return AssociationResult(
            var1=var1.name,
            var2=var2.name,
            association_type=AssociationType.CATEGORICAL_CATEGORICAL,
            method="cramers_v",
            statistic=cramers_v,
            p_value=p_value,
            effect_size=cramers_v,
            sample_size=n,
            interpretation=self._interpret_cramers_v(cramers_v)
        )
    
    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Interpret Cramér's V effect size."""
        if cramers_v < 0.1:
            return "Negligible association"
        elif cramers_v < 0.3:
            return "Weak association"
        elif cramers_v < 0.5:
            return "Moderate association"
        else:
            return "Strong association"


class MixedAssociationAnalyzer(BaseAssociationAnalyzer):
    """Analyzer for numerical-categorical variable associations."""
    
    def can_analyze(self, var1: pd.Series, var2: pd.Series) -> bool:
        """Check if one variable is numerical and one is categorical."""
        var1_numeric = pd.api.types.is_numeric_dtype(var1)
        var2_numeric = pd.api.types.is_numeric_dtype(var2)
        return var1_numeric != var2_numeric  # XOR: exactly one is numeric
    
    def analyze(self, var1: pd.Series, var2: pd.Series) -> AssociationResult:
        """Analyze association between numerical and categorical variables."""
        self._validate_data(var1, var2)
        
        # Identify which is numerical and which is categorical
        if pd.api.types.is_numeric_dtype(var1):
            numerical_var, categorical_var = var1, var2
            num_name, cat_name = var1.name, var2.name
        else:
            numerical_var, categorical_var = var2, var1
            num_name, cat_name = var2.name, var1.name
        
        # Remove missing values
        mask = ~(numerical_var.isna() | categorical_var.isna())
        clean_numerical = numerical_var[mask]
        clean_categorical = categorical_var[mask]
        
        # Group numerical values by categories
        groups = [clean_numerical[clean_categorical == cat] 
                 for cat in clean_categorical.unique()]
        
        # Remove empty groups
        groups = [group for group in groups if len(group) > 0]
        
        if len(groups) < 2:
            raise InsufficientDataError("Need at least 2 non-empty groups for analysis")
        
        # ANOVA F-test
        f_statistic, p_value = f_oneway(*groups)
        
        # Eta-squared (effect size)
        # SS_between / SS_total
        grand_mean = clean_numerical.mean()
        ss_total = ((clean_numerical - grand_mean) ** 2).sum()
        
        ss_between = sum(
            len(group) * (group.mean() - grand_mean) ** 2 
            for group in groups
        )
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return AssociationResult(
            var1=var1.name,
            var2=var2.name,
            association_type=AssociationType.NUMERICAL_CATEGORICAL,
            method="anova_f",
            statistic=f_statistic,
            p_value=p_value,
            effect_size=eta_squared,
            sample_size=len(clean_numerical),
            interpretation=self._interpret_eta_squared(eta_squared)
        )
    
    def _interpret_eta_squared(self, eta_squared: float) -> str:
        """Interpret eta-squared effect size."""
        if eta_squared < 0.01:
            return "Negligible effect"
        elif eta_squared < 0.06:
            return "Small effect"
        elif eta_squared < 0.14:
            return "Medium effect"
        else:
            return "Large effect"


class AssociationMetricsAnalyzer(BaseAnalyzer):
    """Main analyzer for cross-variable associations."""
    
    def __init__(self, config: LEDAConfig):
        super().__init__(config)
        self.analyzers = [
            NumericalAssociationAnalyzer(config),
            CategoricalAssociationAnalyzer(config),
            MixedAssociationAnalyzer(config)
        ]
    
    @property
    def analyzer_name(self) -> str:
        return "association_metrics"
    
    def can_analyze(self, data: pd.DataFrame) -> bool:
        """Check if association analysis can be performed."""
        return len(data.columns) >= 2
    
    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        """Perform comprehensive association analysis."""
        if not self.can_analyze(data):
            raise InvalidDataTypeError("Need at least 2 variables for association analysis")
        
        associations = []
        correlation_matrix = {}
        
        # Analyze all pairs of variables
        columns = data.columns.tolist()
        for i, col1 in enumerate(columns):
            correlation_matrix[col1] = {}
            for j, col2 in enumerate(columns):
                if i >= j:  # Only analyze upper triangle + diagonal
                    if i == j:
                        # Self-correlation is always 1
                        correlation_matrix[col1][col2] = 1.0
                    continue
                
                try:
                    # Find appropriate analyzer
                    analyzer = self._select_analyzer(data[col1], data[col2])
                    if analyzer:
                        result = analyzer.analyze(data[col1], data[col2])
                        associations.append(result)
                        
                        # Store in matrix format
                        correlation_matrix[col1][col2] = result.statistic
                        correlation_matrix[col2] = correlation_matrix.get(col2, {})
                        correlation_matrix[col2][col1] = result.statistic
                    
                except (InsufficientDataError, InvalidDataTypeError) as e:
                    # Skip problematic pairs
                    correlation_matrix[col1][col2] = None
                    correlation_matrix[col2] = correlation_matrix.get(col2, {})
                    correlation_matrix[col2][col1] = None
        
        # Fill diagonal
        for col in columns:
            correlation_matrix[col][col] = 1.0
        
        # Generate summary statistics
        summary = self._generate_summary(associations)
        
        result_data = {
            "associations": [assoc.to_dict() for assoc in associations],
            "correlation_matrix": correlation_matrix,
            "summary": summary,
            "metadata": {
                "total_pairs": len(associations),
                "variable_count": len(columns),
                "analysis_timestamp": pd.Timestamp.now().isoformat()
            }
        }
        
        return AnalysisResult(
            analyzer_name=self.analyzer_name,
            data=result_data,
            metadata={"total_associations": len(associations)}
        )
    
    def _select_analyzer(
        self, var1: pd.Series, var2: pd.Series
    ) -> Optional[BaseAssociationAnalyzer]:
        """Select appropriate analyzer for variable pair."""
        for analyzer in self.analyzers:
            if analyzer.can_analyze(var1, var2):
                return analyzer
        return None
    
    def _generate_summary(self, associations: List[AssociationResult]) -> Dict[str, Any]:
        """Generate summary statistics for associations."""
        if not associations:
            return {
                "total_associations": 0,
                "significant_associations": 0,
                "average_effect_size": 0,
                "strongest_association": None
            }
        
        # Filter significant associations
        alpha = self.config.analysis.significance_level
        significant = [
            a for a in associations 
            if a.p_value is not None and a.p_value < alpha
        ]
        
        # Find strongest association
        strongest = max(
            associations, 
            key=lambda x: abs(x.statistic) if x.statistic is not None else 0
        )
        
        # Calculate average effect sizes by type
        effect_sizes_by_type = {}
        for assoc in associations:
            if assoc.effect_size is not None:
                assoc_type = assoc.association_type.value
                if assoc_type not in effect_sizes_by_type:
                    effect_sizes_by_type[assoc_type] = []
                effect_sizes_by_type[assoc_type].append(assoc.effect_size)
        
        avg_effect_sizes = {
            assoc_type: np.mean(sizes) 
            for assoc_type, sizes in effect_sizes_by_type.items()
        }
        
        return {
            "total_associations": len(associations),
            "significant_associations": len(significant),
            "significance_rate": len(significant) / len(associations),
            "average_effect_sizes": avg_effect_sizes,
            "strongest_association": {
                "variables": f"{strongest.var1} - {strongest.var2}",
                "statistic": strongest.statistic,
                "method": strongest.method,
                "interpretation": strongest.interpretation
            } if strongest else None,
            "association_types": {
                assoc_type.value: len([a for a in associations if a.association_type == assoc_type])
                for assoc_type in AssociationType
            }
        }


# Utility functions for external use
def calculate_correlation_matrix(
    data: pd.DataFrame, 
    method: str = "pearson",
    min_periods: int = None
) -> pd.DataFrame:
    """Calculate correlation matrix with specified method."""
    if method not in ["pearson", "spearman", "kendall"]:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return data.corr(method=method, min_periods=min_periods)


def find_highly_correlated_pairs(
    data: pd.DataFrame, 
    threshold: float = 0.8,
    method: str = "pearson"
) -> List[Tuple[str, str, float]]:
    """Find pairs of variables with correlation above threshold."""
    corr_matrix = calculate_correlation_matrix(data, method=method)
    
    pairs = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Only upper triangle
                correlation = corr_matrix.loc[col1, col2]
                if abs(correlation) >= threshold:
                    pairs.append((col1, col2, correlation))
    
    return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)


def detect_multicollinearity(
    data: pd.DataFrame, 
    vif_threshold: float = 5.0
) -> Dict[str, float]:
    """Detect multicollinearity using Variance Inflation Factor."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
        
        # Only include numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return {}
        
        # Add constant for VIF calculation
        X = add_constant(numeric_data)
        
        vif_data = {}
        for i, col in enumerate(numeric_data.columns):
            vif = variance_inflation_factor(X.values, i + 1)  # +1 for constant
            vif_data[col] = vif
        
        return vif_data
    
    except ImportError:
        # statsmodels not available
        return {}