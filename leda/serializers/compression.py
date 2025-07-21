"""Compression utilities for serialized data."""

from __future__ import annotations

from typing import Optional

from ..config.schemas import CompressionMethod
from ..exceptions.data_exceptions import CompressionError


class CompressionUtility:
    """Utility class for data compression operations."""
    
    @staticmethod
    def compress(
        data: bytes, 
        method: CompressionMethod,
        level: Optional[int] = None
    ) -> bytes:
        """
        Compress data using specified method.
        
        Args:
            data: Raw data to compress
            method: Compression method to use
            level: Compression level (method-specific)
            
        Returns:
            Compressed data
            
        Raises:
            CompressionError: If compression fails
        """
        try:
            if method == CompressionMethod.LZ4:
                return CompressionUtility._compress_lz4(data, level)
            elif method == CompressionMethod.ZSTD:
                return CompressionUtility._compress_zstd(data, level)
            elif method == CompressionMethod.GZIP:
                return CompressionUtility._compress_gzip(data, level)
            elif method == CompressionMethod.NONE:
                return data
            else:
                raise CompressionError(f"Unsupported compression method: {method}")
                
        except Exception as e:
            raise CompressionError(f"Compression failed with {method}: {str(e)}") from e
    
    @staticmethod
    def decompress(
        data: bytes, 
        method: CompressionMethod
    ) -> bytes:
        """
        Decompress data using specified method.
        
        Args:
            data: Compressed data
            method: Compression method used
            
        Returns:
            Decompressed data
            
        Raises:
            CompressionError: If decompression fails
        """
        try:
            if method == CompressionMethod.LZ4:
                return CompressionUtility._decompress_lz4(data)
            elif method == CompressionMethod.ZSTD:
                return CompressionUtility._decompress_zstd(data)
            elif method == CompressionMethod.GZIP:
                return CompressionUtility._decompress_gzip(data)
            elif method == CompressionMethod.NONE:
                return data
            else:
                raise CompressionError(f"Unsupported compression method: {method}")
                
        except Exception as e:
            raise CompressionError(f"Decompression failed with {method}: {str(e)}") from e
    
    @staticmethod
    def _compress_lz4(data: bytes, level: Optional[int] = None) -> bytes:
        """Compress using LZ4."""
        try:
            import lz4.frame
            if level is not None:
                return lz4.frame.compress(data, compression_level=level)
            return lz4.frame.compress(data)
        except ImportError:
            raise CompressionError("lz4 library not available. Install with: pip install lz4")
    
    @staticmethod
    def _decompress_lz4(data: bytes) -> bytes:
        """Decompress LZ4 data."""
        try:
            import lz4.frame
            return lz4.frame.decompress(data)
        except ImportError:
            raise CompressionError("lz4 library not available. Install with: pip install lz4")
    
    @staticmethod
    def _compress_zstd(data: bytes, level: Optional[int] = None) -> bytes:
        """Compress using Zstandard."""
        try:
            import zstandard as zstd
            cctx = zstd.ZstdCompressor(level=level if level is not None else 3)
            return cctx.compress(data)
        except ImportError:
            raise CompressionError("zstandard library not available. Install with: pip install zstandard")
    
    @staticmethod
    def _decompress_zstd(data: bytes) -> bytes:
        """Decompress Zstandard data."""
        try:
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        except ImportError:
            raise CompressionError("zstandard library not available. Install with: pip install zstandard")
    
    @staticmethod
    def _compress_gzip(data: bytes, level: Optional[int] = None) -> bytes:
        """Compress using gzip."""
        import gzip
        return gzip.compress(data, compresslevel=level if level is not None else 6)
    
    @staticmethod
    def _decompress_gzip(data: bytes) -> bytes:
        """Decompress gzip data."""
        import gzip
        return gzip.decompress(data)
    
    @staticmethod
    def estimate_compression_ratio(
        data: bytes, 
        method: CompressionMethod
    ) -> float:
        """
        Estimate compression ratio without full compression.
        
        Args:
            data: Data to estimate compression for
            method: Compression method
            
        Returns:
            Estimated compression ratio (compressed_size / original_size)
        """
        if method == CompressionMethod.NONE:
            return 1.0
        
        # Sample-based estimation for large data
        sample_size = min(len(data), 1024)
        sample_data = data[:sample_size]
        
        try:
            compressed_sample = CompressionUtility.compress(sample_data, method)
            ratio = len(compressed_sample) / len(sample_data)
            return max(0.1, min(1.0, ratio))  # Clamp between 10% and 100%
        except Exception:
            return 0.7  # Conservative estimate

