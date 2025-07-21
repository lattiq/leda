"""Base serializer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Union


class BaseSerializer(ABC):
    """Abstract base class for data serializers."""
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the format name (e.g., 'msgpack', 'json')."""
        pass
    
    @property
    @abstractmethod
    def mime_type(self) -> str:
        """Return the MIME type for this format."""
        pass
    
    @abstractmethod
    def serialize(self, data: Dict[str, Any]) -> Union[bytes, str]:
        """
        Serialize data to the target format.
        
        Args:
            data: Dictionary to serialize
            
        Returns:
            Serialized data as bytes or string
        """
        pass
    
    @abstractmethod
    def deserialize(self, data: Union[bytes, str]) -> Dict[str, Any]:
        """
        Deserialize data from the target format.
        
        Args:
            data: Serialized data
            
        Returns:
            Deserialized dictionary
        """
        pass
    
    def supports_binary(self) -> bool:
        """Return True if this serializer produces binary output."""
        return False
