"""JSON serializer implementation."""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base_serializer import BaseSerializer
from ..config.schemas import SerializationConfig
from ..exceptions import SerializationError


class JSONSerializer(BaseSerializer):
    """
    JSON serializer with numpy/pandas support.
    
    Provides compatibility fallback for systems that don't support MessagePack.
    """
    
    def __init__(self, config: SerializationConfig):
        self.config = config
        self._setup_encoder()
    
    def _setup_encoder(self):
        """Configure JSON encoder for numpy/pandas types."""
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                return super().default(obj)
        
        self._encoder = CustomJSONEncoder
    
    @property
    def format_name(self) -> str:
        return "json"
    
    @property
    def mime_type(self) -> str:
        return "application/json"
    
    def serialize(self, data: Dict[str, Any]) -> str:
        """Serialize data to JSON format."""
        try:
            # Apply precision settings
            processed_data = self._process_numeric_precision(data)
            
            # Use orjson for better performance if available
            try:
                import orjson
                return orjson.dumps(
                    processed_data,
                    option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS
                ).decode('utf-8')
            except ImportError:
                # Fallback to standard json
                return json.dumps(
                    processed_data,
                    cls=self._encoder,
                    ensure_ascii=False,
                    separators=(',', ':')  # Compact representation
                )
                
        except Exception as e:
            raise SerializationError(f"JSON serialization failed: {str(e)}") from e
    
    def deserialize(self, data: str) -> Dict[str, Any]:
        """Deserialize data from JSON format."""
        try:
            try:
                import orjson
                return orjson.loads(data)
            except ImportError:
                return json.loads(data)
                
        except Exception as e:
            raise SerializationError(f"JSON deserialization failed: {str(e)}") from e
    
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

