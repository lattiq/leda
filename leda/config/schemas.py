"""Pydantic schemas for configuration validation."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, validator


class AnalysisConfig(BaseModel):
    """Configuration for data analysis parameters."""
    
    max_unique_values: int = Field(50, ge=1, description="Max unique values for categorical analysis")
    correlation_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Minimum correlation to report")
    outlier_method: Literal["iqr", "zscore", "isolation_forest"] = Field("iqr", description="Outlier detection method")
    missing_threshold: float = Field(0.05, ge=0.0, le=1.0, description="Threshold for missing data warnings")
    sample_size: int | None = Field(None, ge=100, description="Sample size for large datasets")
    enable_streaming: bool = Field(True, description="Enable streaming analysis for large datasets")
    
    class Config:
        extra = "forbid"  # Prevent typos in configuration


class VisualizationConfig(BaseModel):
    """Configuration for visualization generation."""
    
    theme: str = Field("default", description="Visualization theme")
    width: int = Field(800, ge=100, description="Default plot width")
    height: int = Field(600, ge=100, description="Default plot height") 
    color_palette: list[str] = Field(default_factory=lambda: ["#1f77b4", "#ff7f0e", "#2ca02c"], description="Color palette")
    show_distributions: bool = Field(True, description="Generate distribution plots")
    show_correlations: bool = Field(True, description="Generate correlation plots")
    show_missing_patterns: bool = Field(True, description="Generate missing data visualizations")
    max_categories: int = Field(20, ge=1, description="Maximum categories to show in plots")
    
    @field_validator("color_palette")
    @classmethod
    def validate_colors(cls, v):
        """Validate that color palette has at least 2 colors."""
        if not v or len(v) < 2:
            raise ValueError("Color palette must contain at least 2 colors")
        return v

    class Config:
        extra = "forbid"


class CompressionMethod(str, Enum):
    """Supported compression methods."""
    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"
    GZIP = "gzip"

class SerializationConfig(BaseModel):
    """Configuration for data serialization."""
    
    primary_format: Literal["msgpack", "json"] = Field(
        default="msgpack",
        description="Primary serialization format"
    )
    
    enable_compression: bool = Field(
        default=False,
        description="Enable compression for serialized data"
    )
    
    compression_method: CompressionMethod = Field(
        default=CompressionMethod.GZIP,
        description="Compression algorithm to use"
    )
    
    precision: int = Field(
        default=6,
        ge=1,
        le=15,
        description="Decimal precision for floating point numbers"
    )
    
    max_array_size: int = Field(
        default=10000,
        ge=100,
        description="Maximum array size before streaming"
    )
    
    enable_streaming: bool = Field(
        default=False,
        description="Enable streaming for large datasets"
    )
    
    chunk_size: int = Field(
        default=1000,
        ge=100,
        description="Chunk size for streaming serialization"
    )
    
    @field_validator('compression_method')
    @classmethod
    def validate_compression_method(cls, v, values):
        """Validate compression method is available."""
        if not values.data.get('enable_compression', False):
            return v
            
        # Import checks for compression libraries
        if v == CompressionMethod.LZ4:
            try:
                import lz4.frame
            except ImportError:
                raise ValueError("lz4 library not available. Install with: pip install lz4")
        elif v == CompressionMethod.ZSTD:
            try:
                import zstandard
            except ImportError:
                raise ValueError("zstandard library not available. Install with: pip install zstandard")
        
        return v


class OutputConfig(BaseModel):
    """Configuration for output generation."""
    
    formats: list[Literal["msgpack", "json", "html", "pdf"]] = Field(
        default_factory=lambda: ["msgpack"], 
        description="Output formats to generate"
    )
    output_dir: str = Field(".", description="Output directory for reports")
    filename_prefix: str = Field("leda_analysis", description="Filename prefix")
    include_raw_data: bool = Field(False, description="Include raw data in output")
    enable_cdn_optimization: bool = Field(True, description="Optimize for CDN delivery")
    
    class Config:
        extra = "forbid"


class LEDAConfig(BaseModel):
    """Main configuration class combining all settings."""
    
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig, description="Analysis configuration")
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig, description="Visualization configuration") 
    serialization: SerializationConfig = Field(default_factory=SerializationConfig, description="Serialization configuration")
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output configuration")
    
    # Global settings
    verbose: bool = Field(False, description="Enable verbose logging")
    n_jobs: int = Field(-1, description="Number of parallel jobs (-1 for all cores)")
    random_state: int = Field(42, description="Random seed for reproducibility")
    
    class Config:
        extra = "forbid"
        
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation and setup."""
        # Validate format compatibility
        values = self.model_dump()
        if "msgpack" in values["output"]["formats"] and values["serialization"]["primary_format"] != "msgpack":
            values["serialization"]["primary_format"] = "msgpack"
            self = self.model_validate(values)
