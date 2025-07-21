"""Fixed comprehensive tests for LEDA serializers."""

from pathlib import Path
import sys
import pytest
import numpy as np

# Import LEDA components
from leda.config.schemas import SerializationConfig, CompressionMethod
from leda.config.defaults import (
    get_default_serialization_config,
    get_performance_serialization_config,
    get_compatibility_serialization_config,
)
from leda.serializers import (
    FormatNegotiator,
    JSONSerializer,
    CompressionUtility,
)

# Add project root to path for development
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Optional imports for testing
try:
    from leda.serializers import MessagePackSerializer
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

try:
    from leda.serializers import StreamingSerializer
    HAS_STREAMING = True
except ImportError:
    HAS_STREAMING = False


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


class TestJSONSerializer:
    """Test JSON serializer."""
    
    def test_basic_serialization(self, sample_eda_results):
        """Test basic JSON serialization."""
        config = get_default_serialization_config()
        serializer = JSONSerializer(config)
        
        serialized = serializer.serialize(sample_eda_results)
        deserialized = serializer.deserialize(serialized)
        
        assert isinstance(serialized, str)
        assert deserialized == sample_eda_results
    
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
    
    def test_precision_handling(self):
        """Test numeric precision handling."""
        config = SerializationConfig(precision=2)
        serializer = JSONSerializer(config)
        
        data = {"float": 3.14159}
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)
        
        assert deserialized["float"] == 3.14


@pytest.mark.skipif(not HAS_MSGPACK, reason="msgpack not available")
class TestMessagePackSerializer:
    """Test MessagePack serializer."""
    
    def test_basic_serialization(self, sample_eda_results):
        """Test basic MessagePack serialization."""
        config = get_default_serialization_config()
        serializer = MessagePackSerializer(config)
        
        serialized = serializer.serialize(sample_eda_results)
        deserialized = serializer.deserialize(serialized)
        
        assert isinstance(serialized, bytes)
        assert deserialized == sample_eda_results
    
    def test_binary_format_support(self):
        """Test binary format support."""
        config = get_default_serialization_config()
        serializer = MessagePackSerializer(config)
        
        assert serializer.supports_binary() is True
        assert serializer.format_name == "msgpack"
        assert serializer.mime_type == "application/msgpack"


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
    
    def test_serialization_with_format_selection(self, sample_eda_results):
        """Test serialization with format selection."""
        config = get_default_serialization_config()
        negotiator = FormatNegotiator(config)
        
        # Test explicit format selection
        json_data = negotiator.serialize(sample_eda_results, "json")
        assert isinstance(json_data, str)
        
        # Test deserialization
        deserialized = negotiator.deserialize(json_data, "json")
        assert deserialized == sample_eda_results
    
    def test_format_auto_detection(self, sample_eda_results):
        """Test automatic format detection."""
        config = get_default_serialization_config()
        negotiator = FormatNegotiator(config)
        
        # Serialize to JSON
        json_data = negotiator.serialize(sample_eda_results, "json")
        
        # Auto-detect format during deserialization
        deserialized = negotiator.deserialize(json_data)
        assert deserialized == sample_eda_results


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


class TestIntegration:
    """Integration tests for serializer components."""
    
    def test_end_to_end_serialization(self, sample_eda_results):
        """Test complete serialization workflow."""
        config = get_performance_serialization_config()
        negotiator = FormatNegotiator(config)
        
        # Test available formats
        for format_name in negotiator.get_available_formats():
            # Serialize
            serialized = negotiator.serialize(sample_eda_results, format_name)
            
            # Deserialize
            deserialized = negotiator.deserialize(serialized, format_name)
            
            # Verify structure preservation
            assert deserialized["summary"]["shape"]["rows"] == 1000
            assert deserialized["columns"]["age"]["mean"] == 35.2

