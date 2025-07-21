"""Setup script for LEDA development."""

from setuptools import setup, find_packages

# Read version from _version.py
def get_version():
    with open('leda/_version.py', 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return '0.1.0'

setup(
    name='leda',
    version=get_version(),
    description='LattIQ EDA - High-performance exploratory data analysis library',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'plotly>=5.0.0',
        'msgpack>=1.0.0',
        'pydantic>=2.0.0',
        'typing-extensions>=4.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'ruff>=0.1.0',
            'mypy>=1.0.0',
        ],
        'compression': [
            'lz4>=4.0.0',
            'zstandard>=0.20.0',
        ],
        'json': [
            'orjson>=3.8.0',
        ],
    },
)
