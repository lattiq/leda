"""Format negotiation and serialization coordination."""

from __future__ import annotations

from typing import Any, Dict, Union

from .base_serializer import BaseSerializer
from .msgpack_serializer import MessagePackSerializer
from .json_serializer import JSONSerializer
from ..config.schemas import SerializationConfig
from ..exceptions import FormatError


class FormatNegotiator:
    """
    Handles format negotiation and serialization coordination.
    
    Manages MessagePack-first approach with JSON fallback.
    """
    
    def __init__(self, config: SerializationConfig):
        self.config = config
        self._serializers = self._initialize_serializers()
    
    def _initialize_serializers(self) -> Dict[str, BaseSerializer]:
        """Initialize available serializers."""
        serializers = {}
        
        # MessagePack serializer (primary)
        try:
            serializers["msgpack"] = MessagePackSerializer(self.config)
        except ImportError:
            pass  # MessagePack not available
        
        # JSON serializer (fallback)
        serializers["json"] = JSONSerializer(self.config)
        
        return serializers
    
    def serialize(self, data: Dict[str, Any], format_name: str = None) -> Union[bytes, str]:
        """
        Serialize data in the requested format.
        
        Args:
            data: Data to serialize
            format_name: Target format ('msgpack', 'json', or None for auto)
            
        Returns:
            Serialized data
            
        Raises:
            FormatError: If format is not supported
            SerializationError: If serialization fails
        """
        if format_name is None:
            format_name = self.config.primary_format
        
        if format_name not in self._serializers:
            # Fallback to JSON if requested format unavailable
            if "json" in self._serializers:
                format_name = "json"
            else:
                raise FormatError(f"No serializer available for format: {format_name}")
        
        serializer = self._serializers[format_name]
        return serializer.serialize(data)
    
    def deserialize(self, data: Union[bytes, str], format_name: str = None) -> Dict[str, Any]:
        """
        Deserialize data from the specified format.
        
        Args:
            data: Serialized data
            format_name: Source format ('msgpack', 'json', or None for auto-detect)
            
        Returns:
            Deserialized data
            
        Raises:
            FormatError: If format is not supported
            SerializationError: If deserialization fails
        """
        if format_name is None:
            format_name = self._detect_format(data)
        
        if format_name not in self._serializers:
            raise FormatError(f"No deserializer available for format: {format_name}")
        
        serializer = self._serializers[format_name]
        return serializer.deserialize(data)
    
    def _detect_format(self, data: Union[bytes, str]) -> str:
        """Auto-detect data format."""
        if isinstance(data, bytes):
            # Try MessagePack first
            if "msgpack" in self._serializers:
                try:
                    # Quick validation - MessagePack should start with specific byte patterns
                    if data and data[0] in [0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,  # fixmap
                                           0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,  # fixarray
                                           0xa0, 0xde, 0xdf, 0xc0, 0xc2, 0xc3]:  # common MessagePack types
                        return "msgpack"
                except Exception:
                    pass
            
            # Try to decode as UTF-8 for JSON
            try:
                data.decode('utf-8')
                return "json"
            except UnicodeDecodeError:
                pass
        else:
            # String data is likely JSON
            return "json"
        
        # Default fallback
        return self.config.primary_format
    
    def negotiate_format(self, accept_header: str = None) -> str:
        """
        Negotiate optimal format based on Accept header.
        
        Args:
            accept_header: HTTP Accept header value
            
        Returns:
            Optimal format name
        """
        if not accept_header:
            return self.config.primary_format
        
        # Parse Accept header preferences
        formats = []
        for item in accept_header.split(','):
            item = item.strip()
            if 'application/msgpack' in item:
                formats.append('msgpack')
            elif 'application/json' in item:
                formats.append('json')
        
        # Return first available format
        for format_name in formats:
            if format_name in self._serializers:
                return format_name
        
        return self.config.primary_format
    
    def get_available_formats(self) -> list[str]:
        """Get list of available serialization formats."""
        return list(self._serializers.keys())
    
    def get_mime_type(self, format_name: str) -> str:
        """Get MIME type for a format."""
        if format_name in self._serializers:
            return self._serializers[format_name].mime_type
        return "application/octet-stream"
    
    def estimate_size_reduction(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate size reduction for available formats.
        
        Args:
            data: Data to estimate size for
            
        Returns:
            Dictionary mapping format names to estimated size ratios
        """
        estimates = {}
        
        # Get baseline JSON size
        if "json" in self._serializers:
            json_data = self._serializers["json"].serialize(data)
            json_size = len(json_data.encode('utf-8') if isinstance(json_data, str) else json_data)
            estimates["json"] = 1.0
            
            # Estimate MessagePack compression
            if "msgpack" in self._serializers:
                # Quick estimation: MessagePack typically 20-40% smaller
                estimates["msgpack"] = 0.7
                
                # If compression enabled, estimate additional reduction
                if self.config.enable_compression:
                    from .compression import CompressionUtility
                    compression_ratio = CompressionUtility.estimate_compression_ratio(
                        json_data.encode('utf-8') if isinstance(json_data, str) else json_data,
                        self.config.compression_method
                    )
                    estimates["msgpack"] *= compression_ratio
        
        return estimates