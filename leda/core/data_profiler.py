"""Main data profiler orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from leda.config.schemas import LEDAConfig
from leda.core.basic_stats import BasicStatsAnalyzer
from leda.core.numerical_stats import NumericalAnalyzer
from leda.core.categorical_stats import CategoricalAnalyzer
from leda.core.base_analyzer import BaseAnalyzer, AnalysisResult
from leda.exceptions import DataError, UnsupportedDataTypeError
from leda.serializers.format_negotiator import FormatNegotiator
from leda.utils.data_utils import load_data, infer_data_types

logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Main orchestrator for comprehensive data profiling.
    
    Coordinates multiple analyzers and manages the analysis workflow.
    """
    
    def __init__(
        self,
        config: Optional[LEDAConfig] = None,
        custom_analyzers: Optional[List[BaseAnalyzer]] = None,
    ):
        from leda.config import get_default_config
        
        self.config = config or get_default_config()
        self._setup_logging()
        
        # Initialize core analyzers
        self._analyzers = self._initialize_analyzers()
        
        # Add custom analyzers if provided
        if custom_analyzers:
            self._analyzers.extend(custom_analyzers)
        
        # Initialize serialization
        self._format_negotiator = FormatNegotiator(self.config.serialization)
    
    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    def _initialize_analyzers(self) -> List[BaseAnalyzer]:
        """Initialize the default set of analyzers."""
        return [
            BasicStatsAnalyzer(self.config),
            NumericalAnalyzer(self.config),
            CategoricalAnalyzer(self.config),
        ]
    
    def profile(
        self,
        data: Union[pd.DataFrame, str, Path],
        *,
        output_format: str = "msgpack",
        include_visualizations: bool = True,
        columns: Optional[List[str]] = None,
    ) -> Union[bytes, str, Dict[str, Any]]:
        """
        Profile the provided data and return results in specified format.
        
        Args:
            data: DataFrame or path to CSV/Parquet file
            output_format: "msgpack", "json", or "dict"
            include_visualizations: Whether to generate visualization data
            columns: Specific columns to analyze (None for all)
            
        Returns:
            Serialized analysis results in requested format
        """
        logger.info("Starting data profiling")
        
        # Load and validate data
        df = self._load_and_validate_data(data)
        
        # Select columns to analyze
        if columns:
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                raise DataError(f"Columns not found: {missing_cols}")
            df = df[columns]
        
        # Perform analysis
        results = self._perform_analysis(df, include_visualizations)
        
        # Serialize results
        return self._serialize_results(results, output_format)
    
    def _load_and_validate_data(self, data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        """Load and validate input data."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = load_data(data)
        
        if df.empty:
            raise DataError("Dataset is empty")
        
        if len(df.columns) == 0:
            raise DataError("Dataset has no columns")
        
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def _perform_analysis(self, df: pd.DataFrame, include_visualizations: bool) -> Dict[str, Any]:
        """Perform comprehensive analysis on the DataFrame."""
        results = {
            "summary": self._generate_summary(df),
            "columns": {},
            "correlations": {},
            "metadata": {
                "config": self.config.dict(),
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
            }
        }
        
        # Analyze each column
        for column in df.columns:
            logger.debug(f"Analyzing column: {column}")
            column_results = self._analyze_column(df[column])
            results["columns"][column] = column_results
        
        # Cross-column analysis
        if len(df.select_dtypes(include=[float, int]).columns) > 1:
            results["correlations"] = self._analyze_correlations(df)
        
        # Generate visualizations if requested
        if include_visualizations:
            results["visualizations"] = self._generate_visualizations(df)
        
        return results
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate high-level dataset summary."""
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        return {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "column_types": {
                "numeric": len(numeric_cols),
                "categorical": len(categorical_cols), 
                "datetime": len(datetime_cols),
                "other": len(df.columns) - len(numeric_cols) - len(categorical_cols) - len(datetime_cols)
            },
            "memory_usage": {
                "total_bytes": df.memory_usage(deep=True).sum(),
                "per_column": df.memory_usage(deep=True).to_dict()
            },
            "missing_data": {
                "total_missing": df.isnull().sum().sum(),
                "missing_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            }
        }
    
    def _analyze_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a single column using appropriate analyzers."""
        column_results = {}
        
        for analyzer in self._analyzers:
            if analyzer.can_analyze(series):
                try:
                    result = analyzer.analyze(series)
                    column_results[analyzer.analyzer_name] = {
                        "data": result.data,
                        "metadata": result.metadata,
                        "execution_time": result.execution_time
                    }
                except Exception as e:
                    logger.warning(f"Analyzer {analyzer.analyzer_name} failed for column {series.name}: {e}")
                    column_results[analyzer.analyzer_name] = {
                        "error": str(e),
                        "data": {},
                        "metadata": {}
                    }
        
        return column_results