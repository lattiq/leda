"""Test configuration for LEDA."""

import sys
import os
from pathlib import Path

# Add project root to Python path for imports during testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_data():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 70000, 80000, 90000],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'score': [0.85, 0.92, 0.78, 0.95, 0.88],
    })


@pytest.fixture
def sample_eda_results():
    """Sample EDA analysis results."""
    return {
        'summary': {
            'shape': {'rows': 1000, 'columns': 5},
            'missing_percentage': 2.5,
        },
        'columns': {
            'age': {
                'type': 'numerical',
                'mean': 35.2,
                'std': 12.8,
                'missing_count': 5,
            },
            'category': {
                'type': 'categorical',
                'unique_count': 4,
                'missing_count': 20,
            },
        },
    }
