"""Comprehensive tests for LEDA serializers."""

import pytest
import numpy as np
import pandas as pd

from leda.config.schemas import SerializationConfig, CompressionMethod
from leda.config.defaults import (
    get_default_serialization_config,
    get_performance_serialization_config,
    get_compatibility_serialization_config,
)
from leda.serializers import (
    FormatNegotiator,
    MessagePackSerializer,
    JSONSerializer,
    StreamingSerializer,
    CompressionUtility,
)
from leda.exceptions import SerializationError, CompressionError, FormatError


# Helper functions for conditional testing
def _has_msgpack() -> bool:
    """Check if msgpack is available."""
    try:
        import msgpack
        return True
    except ImportError:
        return False


def _has_lz4() -> bool:
    """Check if lz4 is available."""
    try:
        import lz4.frame
        return True
    except ImportError:
        return False


def _has_zstd() -> bool:
    """Check if zstandard is available."""
    try:
        import zstandard
        return True
    except ImportError:
        return False

class TestSerializationConfig:
    """Test serialization configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = get_default_serialization_config()
        assert config.primary_format == "msgpack"
        assert config.enable_compression is False
        assert config.precision == 6
        assert config.enable_streaming is False
    
    def test_performance_config(self):
        """Test performance-optimized configuration."""
        config = get_performance_serialization_config()
        assert config.primary_format == "msgpack"
        assert config.enable_compression is True
        assert config.compression_method == CompressionMethod.LZ4
        assert config.enable_streaming is True
        assert config.chunk_size == 5000
    
    def test_compatibility_config(self):
        """Test compatibility-focused configuration."""
        config = get_compatibility_serialization_config()
        assert config.primary_format == "json"
        assert config.enable_compression is False
        assert config.enable_streaming is False
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test precision bounds
        with pytest.raises(ValueError):
            SerializationConfig(precision=0)
        
        with pytest.raises(ValueError):
            SerializationConfig(precision=16)
        
        # Valid precision
        config = SerializationConfig(precision=8)
        assert config.precision == 8


class TestCompressionUtility:
    """Test compression utility functions."""
    
    def test_no_compression(self):
        """Test no compression pass-through."""
        data = b"test data"
        compressed = CompressionUtility.compress(data, CompressionMethod.NONE)
        decompressed = CompressionUtility.decompress(compressed, CompressionMethod.NONE)
        
        assert compressed == data
        assert decompressed == data
    
    def test_gzip_compression(self):
        """Test gzip compression/decompression."""
        data = b"test data" * 100  # Make it worth compressing
        compressed = CompressionUtility.compress(data, CompressionMethod.GZIP)
        decompressed = CompressionUtility.decompress(compressed, CompressionMethod.GZIP)
        
        assert len(compressed) < len(data)
        assert decompressed == data
    
    @pytest.mark.skipif(
        not _has_lz4(), 
        reason="lz4 not available"
    )
    def test_lz4_compression(self):
        """Test LZ4 compression/decompression."""
        data = b"test data" * 100
        compressed = CompressionUtility.compress(data, CompressionMethod.LZ4)
        decompressed = CompressionUtility.decompress(compressed, CompressionMethod.LZ4)
        
        assert len(compressed) < len(data)
        assert decompressed == data
    
    @pytest.mark.skipif(
        not _has_zstd(), 
        reason="zstandard not available"
    )
    def test_zstd_compression(self):
        """Test Zstandard compression/decompression."""
        data = b"test data" * 100
        compressed = CompressionUtility.compress(data, CompressionMethod.ZSTD)
        decompressed = CompressionUtility.decompress(compressed, CompressionMethod.ZSTD)
        
        assert len(compressed) < len(data)
        assert decompressed == data
    
    def test_compression_error_handling(self):
        """Test compression error handling."""
        with pytest.raises(CompressionError):
            CompressionUtility.compress(b"test", "invalid_method")
        
        with pytest.raises(CompressionError):
            CompressionUtility.decompress(b"test", "invalid_method")
    
    def test_compression_ratio_estimation(self):
        """Test compression ratio estimation."""
        data = b"test data" * 100
        
        ratio = CompressionUtility.estimate_compression_ratio(data, CompressionMethod.GZIP)
        assert 0.1 <= ratio <= 1.0
        
        ratio_none = CompressionUtility.estimate_compression_ratio(data, CompressionMethod.NONE)
        assert ratio_none == 1.0


class TestJSONSerializer:
    """Test JSON serializer."""
    
    def test_basic_serialization(self):
        """Test basic JSON serialization."""
        config = get_default_serialization_config()
        serializer = JSONSerializer(config)
        
        data = {"key": "value", "number": 42}
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)
        
        assert isinstance(serialized, str)
        assert deserialized == data
    
    def test_numpy_serialization(self):
        """Test numpy type serialization."""
        config = get_default_serialization_config()
        serializer = JSONSerializer(config)
        
        data = {
            "int": np.int64(42),
            "float": np.float64(3.14),
            "array": np.array([1, 2, 3]),
        }
        
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)
        
        assert deserialized["int"] == 42
        assert deserialized["float"] == 3.14
        assert deserialized["array"] == [1, 2, 3]
    
    def test_pandas_serialization(self):
        """Test pandas type serialization."""
        config = get_default_serialization_config()
        serializer = JSONSerializer(config)
        
        data = {
            "timestamp": pd.Timestamp("2023-01-01"),
            "series": pd.Series([1, 2, 3], name="test"),
        }
        
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)
        
        assert "2023-01-01" in deserialized["timestamp"]
        assert isinstance(deserialized["series"], dict)
    
    def test_precision_handling(self):
        """Test numeric precision handling."""
        config = SerializationConfig(precision=2)
        serializer = JSONSerializer(config)
        
        data = {"float": 3.14159}
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)
        
        assert deserialized["float"] == 3.14

    def test_serialization_error_handling(self):
        """Test serialization error handling."""
        config = get_default_serialization_config()
        serializer = JSONSerializer(config)
        
        # Create circular reference which JSON cannot serialize
        data = {}
        data["circular"] = data
        
        with pytest.raises(SerializationError):
            serializer.serialize(data)

@pytest.mark.skipif(
    not _has_msgpack(), 
    reason="msgpack not available"
)
class TestMessagePackSerializer:
    """Test MessagePack serializer."""
    
    def test_basic_serialization(self):
        """Test basic MessagePack serialization."""
        config = get_default_serialization_config()
        serializer = MessagePackSerializer(config)
        
        data = {"key": "value", "number": 42}
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)
        
        assert isinstance(serialized, bytes)
        assert deserialized == data
    
    def test_compression_serialization(self):
        """Test MessagePack with compression."""
        config = SerializationConfig(
            enable_compression=True,
            compression_method=CompressionMethod.GZIP
        )
        serializer = MessagePackSerializer(config)
        
        data = {"key": "value" * 100}  # Make it worth compressing
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)
        
        assert isinstance(serialized, bytes)
        assert deserialized == data
    
    def test_binary_format_support(self):
        """Test binary format support."""
        config = get_default_serialization_config()
        serializer = MessagePackSerializer(config)
        
        assert serializer.supports_binary() is True
        assert serializer.format_name == "msgpack"
        assert serializer.mime_type == "application/msgpack"
    
    def test_numpy_serialization(self):
        """Test numpy type serialization."""
        config = get_default_serialization_config()
        serializer = MessagePackSerializer(config)
        
        data = {
            "int": np.int64(42),
            "float": np.float64(3.14),
            "array": np.array([1, 2, 3]),
        }
        
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)
        
        assert deserialized["int"] == 42
        assert deserialized["float"] == 3.14
        assert deserialized["array"] == [1, 2, 3]


class TestFormatNegotiator:
    """Test format negotiation."""
    
    def test_format_negotiation(self):
        """Test format negotiation logic."""
        config = get_default_serialization_config()
        negotiator = FormatNegotiator(config)
        
        # Test available formats
        formats = negotiator.get_available_formats()
        assert "json" in formats
        
        # Test MIME type retrieval
        assert negotiator.get_mime_type("json") == "application/json"
    
    def test_accept_header_negotiation(self):
        """Test HTTP Accept header negotiation."""
        config = get_default_serialization_config()
        negotiator = FormatNegotiator(config)
        
        # Test JSON preference
        format_name = negotiator.negotiate_format("application/json")
        assert format_name == "json"
        
        # Test multiple formats
        format_name = negotiator.negotiate_format("application/msgpack, application/json")
        assert format_name in ["msgpack", "json"]
    
    def test_serialization_with_format_selection(self):
        """Test serialization with format selection."""
        config = get_default_serialization_config()
        negotiator = FormatNegotiator(config)
        
        data = {"key": "value"}
        
        # Test explicit format selection
        json_data = negotiator.serialize(data, "json")
        assert isinstance(json_data, str)
        
        # Test deserialization
        deserialized = negotiator.deserialize(json_data, "json")
        assert deserialized == data
    
    def test_format_auto_detection(self):
        """Test automatic format detection."""
        config = get_default_serialization_config()
        negotiator = FormatNegotiator(config)
        
        data = {"key": "value"}
        
        # Serialize to JSON
        json_data = negotiator.serialize(data, "json")
        
        # Auto-detect format during deserialization
        deserialized = negotiator.deserialize(json_data)
        assert deserialized == data

    def test_size_estimation(self):
        """Test size reduction estimation."""
        config = get_default_serialization_config()
        negotiator = FormatNegotiator(config)
        
        data = {"key": "value" * 100}
        estimates = negotiator.estimate_size_reduction(data)
        
        assert "json" in estimates
        assert estimates["json"] == 1.0


class TestStreamingSerializer:
    """Test streaming serialization."""
    
    def test_streaming_serialization(self):
        """Test basic streaming serialization."""
        config = get_performance_serialization_config()
        streaming_serializer = StreamingSerializer(config)
        
        # Create test data
        large_data = {
            "small_item": "value",
            "large_array": list(range(1000)),
        }
        
        # Stream serialize
        chunks = list(streaming_serializer.stream_serialize(large_data, "json"))
        
        assert len(chunks) > 0
        assert all(chunk.format_name == "json" for chunk in chunks)
        assert chunks[-1].is_final is True
    
    def test_streaming_deserialization(self):
        """Test streaming deserialization."""
        config = get_performance_serialization_config()
        streaming_serializer = StreamingSerializer(config)
        
        # Create test data
        original_data = {
            "small_item": "value",
            "large_array": list(range(500)),
        }
        
        # Stream serialize and deserialize
        chunks = list(streaming_serializer.stream_serialize(original_data, "json"))
        reconstructed = streaming_serializer.stream_deserialize(iter(chunks))
        
        # Verify reconstruction (arrays might be split/rejoined)
        assert "small_item" in reconstructed
        assert reconstructed["small_item"] == "value"
    
    def test_chunk_size_configuration(self):
        """Test chunk size configuration."""
        config = SerializationConfig(
            enable_streaming=True,
            chunk_size=100,
            max_array_size=200
        )
        streaming_serializer = StreamingSerializer(config)
        
        # Create data larger than chunk size
        large_data = {"array": list(range(300))}
        
        chunks = list(streaming_serializer.stream_serialize(large_data, "json"))
        assert len(chunks) > 1


# Integration test data
@pytest.fixture
def sample_eda_data():
    """Sample EDA data for integration testing."""
    return {
        "summary": {
            "shape": {"rows": 1000, "columns": 5},
            "missing_percentage": 2.5,
            "duplicate_rows": 10,
        },
        "columns": {
            "age": {
                "type": "numerical",
                "stats": {
                    "mean": 35.2,
                    "std": 12.8,
                    "min": 18,
                    "max": 65,
                    "quantiles": [25.0, 35.0, 45.0],
                },
                "missing_count": 5,
            },
            "category": {
                "type": "categorical",
                "stats": {
                    "unique_count": 4,
                    "most_frequent": "A",
                    "frequency": {"A": 400, "B": 300, "C": 200, "D": 100},
                },
                "missing_count": 20,
            },
        },
        "correlations": {
            "age_income": 0.75,
            "age_category": 0.12,
        },
    }


class TestIntegration:
    """Integration tests for serializer components."""
    
    def test_end_to_end_serialization(self, sample_eda_data):
        """Test complete serialization workflow."""
        config = get_performance_serialization_config()
        negotiator = FormatNegotiator(config)
        
        # Test both formats
        for format_name in negotiator.get_available_formats():
            # Serialize
            serialized = negotiator.serialize(sample_eda_data, format_name)
            
            # Deserialize
            deserialized = negotiator.deserialize(serialized, format_name)
            
            # Verify structure preservation
            assert deserialized["summary"]["shape"]["rows"] == 1000
            assert deserialized["columns"]["age"]["stats"]["mean"] == 35.2
            assert len(deserialized["correlations"]) == 2
    
    def test_performance_comparison(self, sample_eda_data):
        """Test performance comparison between formats."""
        config = get_default_serialization_config()
        negotiator = FormatNegotiator(config)
        
        # Get size estimates
        estimates = negotiator.estimate_size_reduction(sample_eda_data)
        
        # JSON should be baseline
        if "json" in estimates:
            assert estimates["json"] == 1.0
        
        # MessagePack should be smaller (if available)
        if "msgpack" in estimates:
            assert estimates["msgpack"] < 1.0
    
    def test_streaming_large_dataset(self):
        """Test streaming with large dataset."""
        # Use a smaller chunk size to force chunking
        config = SerializationConfig(
            enable_streaming=True,
            chunk_size=1000,  # Smaller chunk size
            max_array_size=5000,  # Smaller max array size
            primary_format="json"
        )
        
        streaming_serializer = StreamingSerializer(config)
        
        # Create large dataset that will definitely be chunked
        large_dataset = {
            "large_array": list(range(5000)),  # This should be split into 5 chunks of 1000 each
            "metadata": {"source": "test", "version": "1.0"},
        }
        
        # Stream serialize
        chunks = list(streaming_serializer.stream_serialize(large_dataset, "json"))
        
        # Should have multiple chunks due to large_array being split
        assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
        
        # Verify chunk structure
        assert chunks[0].chunk_id == 0
        assert chunks[-1].is_final is True
        assert all(chunk.format_name == "json" for chunk in chunks)
        
        # Stream deserialize
        reconstructed = streaming_serializer.stream_deserialize(iter(chunks))
        
        # Verify metadata preserved
        assert reconstructed["metadata"]["source"] == "test"
        
        # Verify large array was reconstructed correctly
        assert "large_array" in reconstructed
        assert len(reconstructed["large_array"]) == 5000
        assert reconstructed["large_array"][0] == 0
        assert reconstructed["large_array"][-1] == 4999
    
    def test_compression_effectiveness(self, sample_eda_data):
        """Test compression effectiveness."""
        # Test with and without compression
        config_no_compression = SerializationConfig(enable_compression=False)
        config_with_compression = SerializationConfig(
            enable_compression=True,
            compression_method=CompressionMethod.GZIP
        )
        
        negotiator_no_comp = FormatNegotiator(config_no_compression)
        negotiator_with_comp = FormatNegotiator(config_with_compression)
        
        # Serialize with both configurations
        data_no_comp = negotiator_no_comp.serialize(sample_eda_data, "json")
        data_with_comp = negotiator_with_comp.serialize(sample_eda_data, "json")
        
        # Both should deserialize to same result
        result_no_comp = negotiator_no_comp.deserialize(data_no_comp, "json")
        result_with_comp = negotiator_with_comp.deserialize(data_with_comp, "json")
        
        assert result_no_comp == result_with_comp


if __name__ == "__main__":
    pytest.main([__file__])