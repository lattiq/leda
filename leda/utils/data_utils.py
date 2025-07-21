"""Data manipulation and loading utilities."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any, Optional

from leda.exceptions import DataError, UnsupportedDataTypeError


def load_data(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """
    Load data from various sources.
    
    Args:
        source: File path, URL, or DataFrame
        
    Returns:
        pandas DataFrame
        
    Raises:
        DataError: If data cannot be loaded
        UnsupportedDataTypeError: If file format is not supported
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    
    # Convert to Path object
    if isinstance(source, str):
        source = Path(source)
    
    if not source.exists():
        raise DataError(f"File not found: {source}")
    
    # Determine file type and load accordingly
    suffix = source.suffix.lower()
    
    try:
        if suffix == '.csv':
            return _load_csv(source)
        elif suffix in ['.parquet', '.pq']:
            return _load_parquet(source)
        elif suffix in ['.xlsx', '.xls']:
            return _load_excel(source)
        elif suffix == '.json':
            return _load_json(source)
        elif suffix in ['.pkl', '.pickle']:
            return _load_pickle(source)
        else:
            raise UnsupportedDataTypeError(f"Unsupported file format: {suffix}")
    
    except Exception as e:
        raise DataError(f"Failed to load data from {source}: {str(e)}")


def _load_csv(file_path: Path) -> pd.DataFrame:
    """Load CSV file with intelligent parsing."""
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            return pd.read_csv(
                file_path,
                encoding=encoding,
                low_memory=False,
                na_values=['', 'NA', 'N/A', 'NULL', 'null', 'None', 'none', '-'],
                keep_default_na=True,
            )
        except UnicodeDecodeError:
            continue
    
    raise DataError(f"Could not decode CSV file with any of the tried encodings: {encodings}")


def _load_parquet(file_path: Path) -> pd.DataFrame:
    """Load Parquet file."""
    try:
        import pyarrow.parquet as pq
        return pd.read_parquet(file_path)
    except ImportError:
        # Fallback to pandas implementation
        return pd.read_parquet(file_path)


def _load_excel(file_path: Path) -> pd.DataFrame:
    """Load Excel file."""
    return pd.read_excel(file_path, na_values=['', 'NA', 'N/A', 'NULL', 'null', 'None', 'none', '-'])


def _load_json(file_path: Path) -> pd.DataFrame:
    """Load JSON file."""
    return pd.read_json(file_path, orient='auto')


def _load_pickle(file_path: Path) -> pd.DataFrame:
    """Load pickled DataFrame."""
    return pd.read_pickle(file_path)


def infer_data_types(df: pd.DataFrame, sample_size: int = 10000) -> Dict[str, Any]:
    """
    Infer data types and characteristics of DataFrame columns.
    
    Args:
        df: DataFrame to analyze
        sample_size: Number of rows to sample for type inference
        
    Returns:
        Dictionary with type information for each column
    """
    # Sample data for large DataFrames
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df
    
    type_info = {}
    
    for column in df.columns:
        series = sample_df[column]
        
        # Basic pandas dtype
        dtype = series.dtype
        
        # Infer semantic type
        semantic_type = _infer_semantic_type(series)
        
        # Additional characteristics
        characteristics = _analyze_column_characteristics(series)
        
        type_info[column] = {
            'pandas_dtype': str(dtype),
            'semantic_type': semantic_type,
            'characteristics': characteristics,
            'null_count': int(df[column].isnull().sum()),
            'null_percentage': float(df[column].isnull().sum() / len(df) * 100),
        }
    
    return type_info


def _infer_semantic_type(series: pd.Series) -> str:
    """Infer the semantic type of a series."""
    # Remove null values for analysis
    clean_series = series.dropna()
    
    if clean_series.empty:
        return 'empty'
    
    # Check pandas dtype first
    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series):
            return 'integer'
        else:
            return 'float'
    
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    
    elif pd.api.types.is_bool_dtype(series):
        return 'boolean'
    
    elif pd.api.types.is_categorical_dtype(series):
        return 'categorical'
    
    else:
        # For object dtype, infer based on content
        return _infer_object_type(clean_series)


def _infer_object_type(series: pd.Series) -> str:
    """Infer type for object columns based on content."""
    if series.empty:
        return 'empty'
    
    # Sample for performance
    sample = series.head(min(1000, len(series)))
    
    # Convert to string for analysis
    str_sample = sample.astype(str)
    
    # Check for numeric strings
    numeric_count = 0
    date_count = 0
    email_count = 0
    url_count = 0
    
    for value in str_sample:
        # Numeric patterns
        if value.replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '').isdigit():
            numeric_count += 1
        
        # Date patterns (basic)
        elif any(pattern in value for pattern in ['-', '/', ':', 'T']):
            try:
                pd.to_datetime(value)
                date_count += 1
            except:
                pass
        
        # Email pattern
        elif '@' in value and '.' in value:
            email_count += 1
        
        # URL pattern
        elif value.startswith(('http://', 'https://', 'www.')):
            url_count += 1
    
    total_sample = len(str_sample)
    
    # Determine type based on patterns
    if numeric_count / total_sample > 0.8:
        return 'numeric_string'
    elif date_count / total_sample > 0.8:
        return 'date_string'
    elif email_count / total_sample > 0.8:
        return 'email'
    elif url_count / total_sample > 0.8:
        return 'url'
    elif series.nunique() / len(series) > 0.95:
        return 'identifier'
    else:
        return 'text'


def _analyze_column_characteristics(series: pd.Series) -> Dict[str, Any]:
    """Analyze additional characteristics of a column."""
    characteristics = {}
    
    # Uniqueness
    unique_count = series.nunique()
    total_count = len(series)
    characteristics['unique_count'] = unique_count
    characteristics['uniqueness_ratio'] = unique_count / total_count if total_count > 0 else 0
    
    # Cardinality classification
    uniqueness_ratio = characteristics['uniqueness_ratio']
    if uniqueness_ratio > 0.95:
        characteristics['cardinality'] = 'high'
    elif uniqueness_ratio > 0.5:
        characteristics['cardinality'] = 'medium'
    elif uniqueness_ratio > 0.1:
        characteristics['cardinality'] = 'low'
    else:
        characteristics['cardinality'] = 'very_low'
    
    # For string-like data, analyze length patterns
    if pd.api.types.is_object_dtype(series):
        str_series = series.astype(str)
        lengths = str_series.str.len()
        characteristics['string_length'] = {
            'min': int(lengths.min()),
            'max': int(lengths.max()),
            'mean': float(lengths.mean()),
            'std': float(lengths.std()),
        }
    
    return characteristics
