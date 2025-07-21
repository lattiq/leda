"""Default configuration factory."""

from __future__ import annotations

from .schemas import LEDAConfig, SerializationConfig, CompressionMethod


# Default Configuration
def get_default_config() -> LEDAConfig:
    """Get default LEDA configuration with sensible defaults."""
    return LEDAConfig()

# Performance Configuration
def get_performance_config() -> LEDAConfig:
    """Get performance-optimized configuration for large datasets."""
    config = LEDAConfig()
    config.analysis.enable_streaming = True
    config.analysis.sample_size = 50000
    config.serialization.enable_compression = True
    config.serialization.compression_method = "lz4"
    config.visualization.max_categories = 10
    return config

# Comprehensive Configuration
def get_comprehensive_config() -> LEDAConfig:
    """Get comprehensive analysis configuration for detailed reports."""
    config = LEDAConfig()
    config.analysis.max_unique_values = 100
    config.analysis.correlation_threshold = 0.05
    config.visualization.max_categories = 50
    config.output.formats = ["msgpack", "json", "html"]
    config.output.include_raw_data = True
    return config

# Serialization Configurations
def get_default_serialization_config() -> SerializationConfig:
    """Get default serialization configuration."""
    return SerializationConfig()

def get_performance_serialization_config() -> SerializationConfig:
    """Get performance-optimized serialization configuration."""
    return SerializationConfig(
        primary_format="msgpack",
        enable_compression=True,
        compression_method=CompressionMethod.LZ4,
        precision=4,
        enable_streaming=True,
        chunk_size=5000,
        max_array_size=50000,
    )

def get_compatibility_serialization_config() -> SerializationConfig:
    """Get compatibility-focused serialization configuration."""
    return SerializationConfig(
        primary_format="json",
        enable_compression=False,
        precision=6,
        enable_streaming=False,
    )