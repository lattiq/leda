"""Streaming serialization for large datasets."""

from __future__ import annotations

from typing import Any, Dict, Iterator, Union, Optional
from dataclasses import dataclass

from .msgpack_serializer import MessagePackSerializer
from .json_serializer import JSONSerializer
from ..config.schemas import SerializationConfig
from ..exceptions import SerializationError


@dataclass
class StreamChunk:
    """Represents a chunk of streaming data."""
    data: Union[bytes, str]
    chunk_id: int
    is_final: bool
    format_name: str
    total_chunks: Optional[int] = None


class StreamingSerializer:
    """
    Streaming serializer for large datasets.
    
    Breaks large datasets into chunks for memory-efficient processing.
    """
    
    def __init__(self, config: SerializationConfig):
        self.config = config
        self._serializers = {
            "msgpack": MessagePackSerializer(config),
            "json": JSONSerializer(config),
        }
    
    def stream_serialize(
        self, 
        data: Dict[str, Any], 
        format_name: str = "msgpack"
    ) -> Iterator[StreamChunk]:
        """
        Stream serialize large data in chunks.
        
        Args:
            data: Data to serialize
            format_name: Target format
            
        Yields:
            StreamChunk objects containing serialized data
        """
        if format_name not in self._serializers:
            raise SerializationError(f"Unsupported format for streaming: {format_name}")
        
        # Identify large arrays/collections for chunking
        chunks_data = self._chunk_data(data)
        total_chunks = len(chunks_data)
        
        for chunk_id, chunk_data in enumerate(chunks_data):
            try:
                serialized_chunk = self._serializers[format_name].serialize(chunk_data)
                
                yield StreamChunk(
                    data=serialized_chunk,
                    chunk_id=chunk_id,
                    is_final=(chunk_id == total_chunks - 1),
                    format_name=format_name,
                    total_chunks=total_chunks
                )
                
            except Exception as e:
                raise SerializationError(f"Failed to serialize chunk {chunk_id}: {str(e)}") from e
    
    def stream_deserialize(
        self, 
        chunks: Iterator[StreamChunk]
    ) -> Dict[str, Any]:
        """
        Deserialize streaming chunks back to original data.
        
        Args:
            chunks: Iterator of StreamChunk objects
            
        Returns:
            Reconstructed data dictionary
        """
        chunk_buffer = {}
        format_name = None
        
        for chunk in chunks:
            if format_name is None:
                format_name = chunk.format_name
            elif format_name != chunk.format_name:
                raise SerializationError("Inconsistent format across chunks")
            
            if format_name not in self._serializers:
                raise SerializationError(f"Unsupported format: {format_name}")
            
            try:
                chunk_data = self._serializers[format_name].deserialize(chunk.data)
                chunk_buffer[chunk.chunk_id] = chunk_data
                
                if chunk.is_final:
                    break
                    
            except Exception as e:
                raise SerializationError(f"Failed to deserialize chunk {chunk.chunk_id}: {str(e)}") from e
        
        # Reconstruct original data from chunks
        return self._reconstruct_data(chunk_buffer)
    
    def _chunk_data(self, data: Dict[str, Any]) -> list[Dict[str, Any]]:
        """
        Split data into chunks based on array sizes.
        
        Args:
            data: Original data dictionary
            
        Returns:
            List of data chunks
        """
        chunks = []
        current_chunk = {}
        current_size = 0
        
        for key, value in data.items():
            item_size = self._estimate_size(value)
            
            # Check if this single item exceeds max_array_size
            if isinstance(value, list) and len(value) > self.config.chunk_size:
                # If we have a current chunk, finalize it first
                if current_chunk:
                    chunks.append(current_chunk.copy())
                    current_chunk.clear()
                    current_size = 0
                
                # Split the large array into smaller chunks
                for i in range(0, len(value), self.config.chunk_size):
                    chunk_value = value[i:i + self.config.chunk_size]
                    chunk_key = f"{key}_chunk_{i//self.config.chunk_size}"
                    chunks.append({chunk_key: chunk_value})
            
            # Check if adding this item would exceed the max chunk size
            elif current_size + item_size > self.config.max_array_size:
                # Start a new chunk
                if current_chunk:
                    chunks.append(current_chunk.copy())
                current_chunk = {key: value}
                current_size = item_size
            
            else:
                # Add to current chunk
                current_chunk[key] = value
                current_size += item_size
        
        # Add final chunk if it has data
        if current_chunk:
            chunks.append(current_chunk)
        
        # If no chunks were created, return the original data as a single chunk
        return chunks if chunks else [data]
    
    def _reconstruct_data(self, chunk_buffer: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reconstruct original data from chunks.
        
        Args:
            chunk_buffer: Dictionary of chunk_id -> chunk_data
            
        Returns:
            Reconstructed data dictionary
        """
        result = {}
        
        # Sort chunks by ID
        sorted_chunks = sorted(chunk_buffer.items())
        
        for chunk_id, chunk_data in sorted_chunks:
            for key, value in chunk_data.items():
                # Handle chunked arrays
                if "_chunk_" in key:
                    base_key = key.split("_chunk_")[0]
                    if base_key not in result:
                        result[base_key] = []
                    
                    if isinstance(value, list):
                        result[base_key].extend(value)
                    else:
                        result[base_key].append(value)
                else:
                    result[key] = value
        
        return result
    
    def _estimate_size(self, value: Any) -> int:
        """
        Estimate memory size of a value more accurately.
        
        Args:
            value: Value to estimate size for
            
        Returns:
            Estimated size (rough approximation)
        """
        if isinstance(value, list):
            # For lists, consider both length and content
            base_size = len(value)
            # Add rough estimate for content complexity
            if value and isinstance(value[0], (int, float)):
                return base_size * 8  # 8 bytes per number roughly
            elif value and isinstance(value[0], str):
                # Estimate string sizes
                avg_str_len = sum(len(str(item)) for item in value[:10]) // min(10, len(value))
                return base_size * avg_str_len
            else:
                return base_size * 32  # Conservative estimate for complex objects
        elif isinstance(value, dict):
            return len(value) * 64  # Conservative estimate for dict entries
        elif isinstance(value, str):
            return len(value)
        else:
            return 64  # Default size for other objects
