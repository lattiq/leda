"""Input/output utility functions."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Union


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Create a safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Safe filename string
    """
    # Remove or replace problematic characters
    safe_chars = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    safe_chars = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', safe_chars)
    
    # Limit length
    if len(safe_chars) > max_length:
        name, ext = os.path.splitext(safe_chars)
        max_name_length = max_length - len(ext)
        safe_chars = name[:max_name_length] + ext
    
    # Ensure it's not empty or just dots
    if not safe_chars or safe_chars.strip('.'):
        safe_chars = 'unnamed_file'
    
    return safe_chars


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    path = Path(file_path)
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0


def create_temp_filename(prefix: str = 'leda_', suffix: str = '.tmp') -> str:
    """
    Create a temporary filename.
    
    Args:
        prefix: Filename prefix
        suffix: Filename suffix
        
    Returns:
        Temporary filename
    """
    import uuid
    return f"{prefix}{uuid.uuid4().hex}{suffix}"
