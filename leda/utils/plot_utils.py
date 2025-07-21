"""Plotting utility functions."""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import numpy as np


def generate_color_palette(n_colors: int, palette_name: str = 'default') -> List[str]:
    """
    Generate a color palette with specified number of colors.
    
    Args:
        n_colors: Number of colors needed
        palette_name: Name of the color palette
        
    Returns:
        List of color hex codes
    """
    palettes = {
        'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'viridis': ['#440154', '#482777', '#3f4a8a', '#31678e', '#26838f',
                   '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825'],
        'accessible': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    }
    
    base_palette = palettes.get(palette_name, palettes['default'])
    
    if n_colors <= len(base_palette):
        return base_palette[:n_colors]
    else:
        # Extend palette by interpolating colors
        return _extend_palette(base_palette, n_colors)


def _extend_palette(base_palette: List[str], n_colors: int) -> List[str]:
    """Extend a color palette to the required number of colors."""
    import colorsys
    
    extended_palette = base_palette.copy()
    
    while len(extended_palette) < n_colors:
        # Generate intermediate colors
        for i in range(len(base_palette) - 1):
            if len(extended_palette) >= n_colors:
                break
            
            color1 = base_palette[i]
            color2 = base_palette[i + 1]
            intermediate = _interpolate_colors(color1, color2, 0.5)
            extended_palette.append(intermediate)
    
    return extended_palette[:n_colors]


def _interpolate_colors(color1: str, color2: str, t: float) -> str:
    """Interpolate between two hex colors."""
    # Convert hex to RGB
    r1, g1, b1 = _hex_to_rgb(color1)
    r2, g2, b2 = _hex_to_rgb(color2)
    
    # Interpolate
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    
    # Convert back to hex
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def calculate_histogram_bins(data: np.ndarray, method: str = 'auto') -> int:
    """
    Calculate optimal number of histogram bins.
    
    Args:
        data: Numeric data array
        method: Binning method ('auto', 'sturges', 'scott', 'freedman')
        
    Returns:
        Number of bins
    """
    if len(data) == 0:
        return 10
    
    if method == 'sturges':
        return int(np.ceil(np.log2(len(data)) + 1))
    elif method == 'scott':
        h = 3.5 * np.std(data) / (len(data) ** (1/3))
        return int(np.ceil((np.max(data) - np.min(data)) / h))
    elif method == 'freedman':
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        if iqr == 0:
            return int(np.ceil(np.sqrt(len(data))))
        h = 2 * iqr / (len(data) ** (1/3))
        return int(np.ceil((np.max(data) - np.min(data)) / h))
    else:  # auto
        return min(50, max(10, int(np.sqrt(len(data)))))


def format_large_numbers(value: float) -> str:
    """
    Format large numbers with appropriate suffixes.
    
    Args:
        value: Numeric value to format
        
    Returns:
        Formatted string
    """
    if abs(value) >= 1e9:
        return f"{value / 1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.1f}K"
    else:
        return f"{value:.1f}"


def is_color_accessible(color: str, background: str = '#ffffff') -> bool:
    """
    Check if a color provides sufficient contrast for accessibility.
    
    Args:
        color: Foreground color (hex)
        background: Background color (hex)
        
    Returns:
        True if contrast ratio meets WCAG AA standards
    """
    def get_luminance(hex_color: str) -> float:
        """Calculate relative luminance of a color."""
        r, g, b = _hex_to_rgb(hex_color)
        
        # Convert to sRGB
        r, g, b = r/255.0, g/255.0, b/255.0
        
        # Apply gamma correction
        def gamma_correct(c):
            return c/12.92 if c <= 0.03928 else ((c + 0.055)/1.055) ** 2.4
        
        r = gamma_correct(r)
        g = gamma_correct(g)
        b = gamma_correct(b)
        
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    lum1 = get_luminance(color)
    lum2 = get_luminance(background)
    
    # Calculate contrast ratio
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    contrast_ratio = (lighter + 0.05) / (darker + 0.05)
    
    # WCAG AA standard requires 4.5:1 for normal text
    return contrast_ratio >= 4.5