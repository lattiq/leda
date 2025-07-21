"""Fixed basic usage example for LEDA serializers."""

import sys
from pathlib import Path

from leda.config.defaults import (
    get_default_serialization_config,
)
from leda.serializers import FormatNegotiator

# Add project root to path for development
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Demonstrate basic serializer usage."""
    print("LEDA Serializers - Basic Usage Example")
    print("=" * 40)
    
    # Create sample data
    sample_data = {
        'metadata': {
            'version': '1.0.0',
            'timestamp': '2024-01-01T00:00:00Z',
        },
        'analysis': {
            'shape': {'rows': 1000, 'columns': 5},
            'missing_percentage': 2.5,
        },
        'statistics': {
            'mean_age': 35.2,
            'std_income': 15000.0,
            'correlations': [0.65, -0.12, 0.78],
        }
    }
    
    print(f"Sample data keys: {list(sample_data.keys())}")
    
    # Test with default configuration
    config = get_default_serialization_config()
    negotiator = FormatNegotiator(config)
    
    print(f"\nAvailable formats: {negotiator.get_available_formats()}")
    
    # Test serialization
    for format_name in negotiator.get_available_formats():
        try:
            print(f"\nTesting {format_name.upper()} format:")
            
            # Serialize
            serialized = negotiator.serialize(sample_data, format_name)
            size = len(serialized) if isinstance(serialized, bytes) else len(serialized.encode())
            print(f"  Serialized size: {size:,} bytes")
            
            # Deserialize
            deserialized = negotiator.deserialize(serialized, format_name)
            print(f"  Deserialization successful: {deserialized['analysis']['shape']['rows']} rows")
            
        except Exception as e:
            print(f"  Error with {format_name}: {e}")
    
    print("\nBasic usage example completed!")


if __name__ == "__main__":
    main()

