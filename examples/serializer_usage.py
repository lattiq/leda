"""
Comprehensive example of LEDA serializer usage.

This example demonstrates:
1. Basic serialization with different formats
2. Performance optimization with compression
3. Streaming for large datasets
4. Format negotiation for web APIs
5. Error handling and fallbacks
"""

from typing import Dict, Any
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
    StreamingSerializer,
    CompressionUtility,
)


def create_sample_eda_results() -> Dict[str, Any]:
    """Create sample EDA analysis results."""
    return {
        "metadata": {
            "analysis_timestamp": "2024-01-15T10:30:00Z",
            "dataset_name": "sample_data.csv",
            "analysis_version": "1.0.0",
            "total_processing_time": 12.5,
        },
        "summary": {
            "shape": {"rows": 10000, "columns": 15},
            "memory_usage": "2.3 MB",
            "missing_data": {
                "total_missing": 150,
                "missing_percentage": 0.1,
                "columns_with_missing": ["age", "income", "category"],
            },
            "data_quality_score": 0.95,
        },
        "column_analysis": {
            "numerical_columns": {
                "age": {
                    "type": "int64",
                    "statistics": {
                        "count": 9995,
                        "mean": 34.2,
                        "std": 12.8,
                        "min": 18,
                        "25%": 25.0,
                        "50%": 34.0,
                        "75%": 43.0,
                        "max": 65,
                    },
                    "missing_count": 5,
                    "outliers": [67, 68, 70],
                    "distribution": "normal",
                    "skewness": 0.15,
                    "kurtosis": -0.8,
                },
                "income": {
                    "type": "float64",
                    "statistics": {
                        "count": 9900,
                        "mean": 52000.0,
                        "std": 15000.0,
                        "min": 25000.0,
                        "25%": 42000.0,
                        "50%": 51000.0,
                        "75%": 62000.0,
                        "max": 95000.0,
                    },
                    "missing_count": 100,
                    "outliers": [120000, 125000, 150000],
                    "distribution": "log-normal",
                    "skewness": 0.75,
                    "kurtosis": 2.1,
                },
            },
            "categorical_columns": {
                "category": {
                    "type": "object",
                    "unique_count": 5,
                    "most_frequent": "Premium",
                    "frequency_distribution": {
                        "Premium": 3500,
                        "Standard": 3000,
                        "Basic": 2000,
                        "Enterprise": 1000,
                        "Trial": 500,
                    },
                    "missing_count": 45,
                    "cardinality": "low",
                },
                "region": {
                    "type": "object",
                    "unique_count": 12,
                    "most_frequent": "North",
                    "frequency_distribution": {
                        "North": 2500,
                        "South": 2200,
                        "East": 2000,
                        "West": 1800,
                        "Northeast": 800,
                        "Southeast": 700,
                    },
                    "missing_count": 0,
                    "cardinality": "medium",
                },
            },
        },
        "correlations": {
            "age_income": 0.65,
            "age_category": 0.12,
            "income_region": -0.08,
        },
        "visualization_data": {
            "histograms": {
                "age": {
                    "bins": list(range(18, 66, 2)),
                    "counts": [45, 123, 234, 345, 456, 543, 432, 321, 234, 123, 67, 34, 12, 5, 2],
                },
                "income": {
                    "bins": list(range(25000, 100000, 5000)),
                    "counts": [12, 45, 123, 234, 345, 456, 543, 432, 321, 234, 123, 67, 34, 12, 5],
                },
            },
            "scatter_plots": {
                "age_vs_income": {
                    "x_values": np.random.normal(34, 12, 100).tolist(),
                    "y_values": np.random.normal(52000, 15000, 100).tolist(),
                },
            },
        },
    }


def basic_serialization_example():
    """Demonstrate basic serialization functionality."""
    print("=== Basic Serialization Example ===")
    
    # Create sample data
    data = create_sample_eda_results()
    
    # Create format negotiator with default config
    config = get_default_serialization_config()
    negotiator = FormatNegotiator(config)
    
    print(f"Available formats: {negotiator.get_available_formats()}")
    
    # Serialize to different formats
    for format_name in negotiator.get_available_formats():
        try:
            print(f"\nTesting {format_name.upper()} format:")
            
            # Serialize
            serialized = negotiator.serialize(data, format_name)
            print(f"  Serialized size: {len(serialized)} {'bytes' if isinstance(serialized, bytes) else 'characters'}")
            print(f"  MIME type: {negotiator.get_mime_type(format_name)}")
            
            # Deserialize
            deserialized = negotiator.deserialize(serialized, format_name)
            print(f"  Deserialization successful: {deserialized['summary']['shape']['rows']} rows")
            
        except Exception as e:
            print(f"  Error with {format_name}: {e}")


def performance_optimization_example():
    """Demonstrate performance optimization techniques."""
    print("\n=== Performance Optimization Example ===")
    
    data = create_sample_eda_results()
    
    # Test different configurations
    configs = {
        "default": get_default_serialization_config(),
        "performance": get_performance_serialization_config(),
        "compatibility": get_compatibility_serialization_config(),
    }
    
    for config_name, config in configs.items():
        print(f"\n{config_name.upper()} Configuration:")
        print(f"  Primary format: {config.primary_format}")
        print(f"  Compression: {config.enable_compression}")
        if config.enable_compression:
            print(f"  Compression method: {config.compression_method}")
        
        negotiator = FormatNegotiator(config)
        
        # Test serialization
        try:
            serialized = negotiator.serialize(data)
            size = len(serialized) if isinstance(serialized, bytes) else len(serialized.encode())
            print(f"  Serialized size: {size:,} bytes")
            
            # Test size estimation
            estimates = negotiator.estimate_size_reduction(data)
            print(f"  Size estimates: {estimates}")
            
        except Exception as e:
            print(f"  Error: {e}")


def streaming_example():
    """Demonstrate streaming serialization for large datasets."""
    print("\n=== Streaming Serialization Example ===")
    
    # Create larger dataset
    large_data = create_sample_eda_results()
    
    # Add large arrays to simulate real EDA results
    large_data["raw_data_sample"] = {
        "values": list(range(5000)),  # Large array
        "timestamps": pd.date_range("2023-01-01", periods=5000, freq="H").tolist(),
        "categories": np.random.choice(["A", "B", "C", "D"], 5000).tolist(),
    }
    
    # Configure for streaming
    config = get_performance_serialization_config()
    streaming_serializer = StreamingSerializer(config)
    
    print(f"Original data estimated size: ~{len(str(large_data)):,} characters")
    print(f"Chunk size: {config.chunk_size}")
    print(f"Max array size: {config.max_array_size}")
    
    # Stream serialize
    print("\nStreaming serialization:")
    chunks = list(streaming_serializer.stream_serialize(large_data, "json"))
    
    total_size = sum(len(chunk.data) for chunk in chunks)
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Total serialized size: {total_size:,} bytes")
    print(f"  Average chunk size: {total_size // len(chunks):,} bytes")
    
    # Show chunk information
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"  Chunk {i}: {len(chunk.data):,} bytes, final: {chunk.is_final}")
    
    if len(chunks) > 3:
        print(f"  ... and {len(chunks) - 3} more chunks")
    
    # Stream deserialize
    print("\nStreaming deserialization:")
    try:
        reconstructed = streaming_serializer.stream_deserialize(iter(chunks))
        print(f"  Reconstruction successful!")
        print(f"  Reconstructed keys: {list(reconstructed.keys())}")
        
        # Verify some data integrity
        if "summary" in reconstructed:
            print(f"  Sample verification - rows: {reconstructed['summary']['shape']['rows']}")
        
    except Exception as e:
        print(f"  Reconstruction error: {e}")


def format_negotiation_example():
    """Demonstrate format negotiation for web APIs."""
    print("\n=== Format Negotiation Example ===")
    
    data = create_sample_eda_results()
    config = get_default_serialization_config()
    negotiator = FormatNegotiator(config)
    
    # Simulate different client Accept headers
    test_headers = [
        "application/json",
        "application/msgpack",
        "application/msgpack, application/json",
        "application/json, application/msgpack",
        "text/html, application/json",
        "*/*",
        None,
    ]
    
    print("Format negotiation based on Accept headers:")
    
    for header in test_headers:
        negotiated_format = negotiator.negotiate_format(header)
        print(f"  Accept: {header or 'None'}")
        print(f"    -> Negotiated format: {negotiated_format}")
        print(f"    -> MIME type: {negotiator.get_mime_type(negotiated_format)}")
        
        # Test actual serialization
        try:
            serialized = negotiator.serialize(data, negotiated_format)
            size = len(serialized) if isinstance(serialized, bytes) else len(serialized.encode())
            print(f"    -> Size: {size:,} bytes")
        except Exception as e:
            print(f"    -> Error: {e}")
        print()


def compression_comparison_example():
    """Demonstrate compression effectiveness."""
    print("\n=== Compression Comparison Example ===")
    
    data = create_sample_eda_results()
    
    # Test different compression methods
    compression_methods = [
        CompressionMethod.NONE,
        CompressionMethod.GZIP,
        CompressionMethod.LZ4,
        CompressionMethod.ZSTD,
    ]

    config = get_default_serialization_config()
    negotiator = FormatNegotiator(config)
    
    # Create test data for compression
    test_data = negotiator.serialize(data, "json").encode('utf-8')
    original_size = len(test_data)
    
    print(f"Original data size: {original_size:,} bytes")
    print("\nCompression comparison:")
    
    for method in compression_methods:
        try:
            if method == CompressionMethod.NONE:
                compressed = test_data
                ratio = 1.0
                print(f"  {method.value.upper()}: {len(compressed):,} bytes (ratio: {ratio:.3f})")
            else:
                compressed = CompressionUtility.compress(test_data, method)
                decompressed = CompressionUtility.decompress(compressed, method)
                ratio = len(compressed) / original_size
                
                # Verify integrity
                integrity_ok = decompressed == test_data
                
                print(f"  {method.value.upper()}: {len(compressed):,} bytes (ratio: {ratio:.3f}) {'✓' if integrity_ok else '✗'}")
                
        except Exception as e:
            print(f"  {method.value.upper()}: Error - {e}")


def error_handling_example():
    """Demonstrate error handling and fallbacks."""
    print("\n=== Error Handling Example ===")
    
    config = get_default_serialization_config()
    negotiator = FormatNegotiator(config)
    
    # Test various error conditions
    print("Testing error conditions:")
    
    # 1. Invalid format
    try:
        data = {"test": "data"}
        negotiator.serialize(data, "invalid_format")
    except Exception as e:
        print(f"  Invalid format error: {type(e).__name__}: {e}")
    
    # 2. Invalid data for deserialization
    try:
        negotiator.deserialize("invalid json data", "json")
    except Exception as e:
        print(f"  Invalid JSON error: {type(e).__name__}: {e}")
    
    # 3. Compression without library
    try:
        # This might work if libraries are available
        config_with_compression = SerializationConfig(
            enable_compression=True,
            compression_method=CompressionMethod.LZ4
        )
        negotiator_comp = FormatNegotiator(config_with_compression)
        data = {"test": "data"}
        result = negotiator_comp.serialize(data, "json")
        print(f"  Compression test: Success ({len(result)} bytes)")
    except Exception as e:
        print(f"  Compression error: {type(e).__name__}: {e}")
    
    # 4. Automatic fallback demonstration
    print("\nFallback behavior:")
    try:
        # Try to serialize with preferred format
        data = {"test": "fallback"}
        result = negotiator.serialize(data)  # Uses default format
        print(f"  Default serialization: Success with {negotiator.config.primary_format}")
        
        # Test format detection
        detected_format = negotiator._detect_format(result)
        print(f"  Format detection: {detected_format}")
        
    except Exception as e:
        print(f"  Fallback error: {e}")


def web_api_simulation():
    """Simulate web API usage patterns."""
    print("\n=== Web API Simulation ===")
    
    data = create_sample_eda_results()
    
    # Simulate different client scenarios
    scenarios = [
        {
            "name": "Modern Web Browser",
            "accept": "application/json, text/html",
            "description": "Standard web browser request"
        },
        {
            "name": "Mobile App",
            "accept": "application/msgpack, application/json",
            "description": "Bandwidth-conscious mobile application"
        },
        {
            "name": "Python Client",
            "accept": "application/msgpack",
            "description": "Python client with msgpack support"
        },
        {
            "name": "Legacy System",
            "accept": "application/json",
            "description": "Legacy system requiring JSON"
        },
    ]
    
    config = get_performance_serialization_config()
    negotiator = FormatNegotiator(config)
    
    print("API Response Simulation:")
    
    for scenario in scenarios:
        print(f"\n{scenario['name']} ({scenario['description']}):")
        
        # Negotiate format
        format_name = negotiator.negotiate_format(scenario['accept'])
        
        # Serialize response
        try:
            response_data = negotiator.serialize(data, format_name)
            size = len(response_data) if isinstance(response_data, bytes) else len(response_data.encode())
            
            print(f"  Format: {format_name}")
            print(f"  MIME Type: {negotiator.get_mime_type(format_name)}")
            print(f"  Response Size: {size:,} bytes")
            
            # Simulate response headers
            headers = {
                "Content-Type": negotiator.get_mime_type(format_name),
                "Content-Length": str(size),
            }
            
            if config.enable_compression:
                headers["Content-Encoding"] = config.compression_method.value
            
            print(f"  Headers: {headers}")
            
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Run all serialization examples."""
    print("LEDA Serializer Usage Examples")
    print("=" * 50)
    
    try:
        basic_serialization_example()
        performance_optimization_example()
        streaming_example()
        format_negotiation_example()
        compression_comparison_example()
        error_handling_example()
        web_api_simulation()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nExample execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()