"""MessagePack serializer implementation."""

from __future__ import annotations
from typing import Any, Dict

import msgpack
import numpy as np
import pandas as pd

from .base_serializer import BaseSerializer
from .compression import CompressionUtility
from ..config.schemas import SerializationConfig
from ..exceptions import SerializationError


class MessagePackSerializer(BaseSerializer):
    """MessagePack serializer with numpy/pandas support."""
    
    def __init__(self, config: SerializationConfig):
        self.config = config
    
    @property
    def format_name(self) -> str:
        return "msgpack"
    
    @property
    def mime_type(self) -> str:
        return "application/msgpack"
    
    def supports_binary(self) -> bool:
        return True
    
    def serialize(self, data: Dict[str, Any]) -> bytes:
        """Serialize data to MessagePack format."""
        try:
            processed_data = self._process_numeric_precision(data)
            serializable_data = self._make_serializable(processed_data)
            
            # Simple msgpack serialization
            packed_data = msgpack.packb(serializable_data, use_bin_type=True)
            
            if self.config.enable_compression:
                packed_data = CompressionUtility.compress(
                    packed_data, self.config.compression_method
                )
            
            return packed_data
        except Exception as e:
            raise SerializationError(f"MessagePack serialization failed: {str(e)}") from e
    
    def deserialize(self, data: bytes) -> Dict[str, Any]:
        """Deserialize data from MessagePack format."""
        try:
            if self.config.enable_compression:
                data = CompressionUtility.decompress(
                    data, self.config.compression_method
                )
            
            # Simple msgpack deserialization
            return msgpack.unpackb(data, strict_map_key=False)
        except Exception as e:
            raise SerializationError(f"MessagePack deserialization failed: {str(e)}") from e
    
    def _make_serializable(self, data: Any) -> Any:
        """Convert data to msgpack-serializable format."""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, tuple):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, pd.Timestamp):
            return data.isoformat()
        elif isinstance(data, pd.Series):
            return data.to_dict()
        elif isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        elif hasattr(data, '__dict__'):
            return self._make_serializable(data.__dict__)
        else:
            return data
    
    def _process_numeric_precision(self, data: Any) -> Any:
        """Apply numeric precision settings recursively."""
        if isinstance(data, dict):
            return {k: self._process_numeric_precision(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._process_numeric_precision(item) for item in data]
        elif isinstance(data, float):
            return round(data, self.config.precision)
        else:
            return data