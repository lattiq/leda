"""Simple test runner for development."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_tests():
    """Run tests with pytest."""
    try:
        import pytest
        
        # Run tests with coverage if available
        try:
            import pytest_cov
            exit_code = pytest.main([
                "tests/",
                "--cov=leda",
                "--cov-report=term-missing",
                "-v"
            ])
        except ImportError:
            exit_code = pytest.main([
                "tests/",
                "-v"
            ])
        
        return exit_code
        
    except ImportError:
        print("pytest not installed. Install with: pip install pytest")
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)