#!/usr/bin/env python3
"""
Example: How to get basic stats from a dataset using LEDA
"""

import pandas as pd
from leda.core.basic_stats import BasicStatsAnalyzer
from leda.core.numerical_stats import NumericalAnalyzer
from leda.core.categorical_stats import CategoricalAnalyzer
from leda.config.defaults import get_default_config


def get_basic_stats(data):
    """
    Get comprehensive basic statistics for a pandas DataFrame.
    
    Args:
        data: pandas DataFrame or path to CSV file
        
    Returns:
        Dictionary with statistics for each column
    """
    # Load data if it's a file path
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()
    
    # Initialize analyzers
    config = get_default_config()
    basic_analyzer = BasicStatsAnalyzer(config)
    numerical_analyzer = NumericalAnalyzer(config)
    categorical_analyzer = CategoricalAnalyzer(config)
    
    results = {}
    
    print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("=" * 60)
    
    for column in df.columns:
        print(f"\n📊 {column.upper()}")
        print("-" * 40)
        
        # Get basic stats (works for all column types)
        basic_result = basic_analyzer.analyze(df[column])
        basic_data = basic_result.data
        
        # Extract key information
        type_info = basic_data['type_info']
        count_info = basic_data['count']
        uniqueness = basic_data['uniqueness']
        percentages = basic_data['percentages']
        
        # Display basic information
        print(f"Type: {type_info['category']} ({type_info['subcategory']})")
        print(f"Valid: {count_info['valid']:,} / {count_info['total']:,} ({percentages['valid']:.1f}%)")
        
        if count_info['missing'] > 0:
            print(f"Missing: {count_info['missing']:,} ({percentages['missing']:.1f}%)")
        
        print(f"Unique values: {uniqueness['unique_count']:,} ({uniqueness['category']})")
        
        # Store basic results
        results[column] = {
            'basic_stats': basic_data,
            'type': type_info['category']
        }
        
        # Add numerical statistics for numeric columns
        if numerical_analyzer.can_analyze(df[column]):
            num_result = numerical_analyzer.analyze(df[column])
            num_data = num_result.data
            
            descriptive = num_data.get('descriptive_stats', {})
            if descriptive:
                print(f"Range: {descriptive.get('min', 'N/A')} → {descriptive.get('max', 'N/A')}")
                print(f"Mean: {descriptive.get('mean', 0):.2f}")
                print(f"Median: {descriptive.get('median', 0):.2f}")
                print(f"Std Dev: {descriptive.get('std', 0):.2f}")
                
                # Check for outliers
                outliers = num_data.get('outlier_detection', {})
                if outliers and 'iqr' in outliers:
                    outlier_count = len(outliers['iqr'].get('outlier_indices', []))
                    if outlier_count > 0:
                        print(f"⚠️  Outliers detected: {outlier_count}")
            
            results[column]['numerical_stats'] = num_data
        
        # Add categorical statistics for text/categorical columns  
        elif categorical_analyzer.can_analyze(df[column]):
            try:
                cat_result = categorical_analyzer.analyze(df[column])
                cat_data = cat_result.data
                
                # Show top categories
                freq_data = cat_data.get('frequency_analysis', {})
                if freq_data and 'value_counts' in freq_data:
                    top_values = freq_data['value_counts']
                    print(f"Top values: {list(top_values.keys())[:3]}")
                    
                # Show diversity metrics
                diversity = cat_data.get('diversity_metrics', {})
                if diversity and 'shannon_entropy' in diversity:
                    print(f"Shannon entropy: {diversity['shannon_entropy']:.2f}")
                
                results[column]['categorical_stats'] = cat_data
            except Exception as e:
                print(f"⚠️  Categorical analysis failed: {e}")
    
    return results


def main():
    """Example usage with sample data."""
    
    # Create sample dataset
    sample_data = pd.DataFrame({
        'employee_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
        'age': [25, 30, 35, 40, 45, 28, None, 33],
        'salary': [50000, 60000, 70000, 80000, 90000, 55000, 75000, 65000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'HR', 'IT', 'Finance', 'IT'],
        'years_exp': [2, 5, 8, 12, 15, 3, 9, 6],
        'performance': ['Good', 'Excellent', 'Good', 'Outstanding', 'Good', 'Fair', None, 'Good']
    })
    
    print("🚀 LEDA Basic Statistics Example")
    print("=" * 60)
    
    # Get comprehensive statistics
    stats = get_basic_stats(sample_data)
    
    print("\n" + "=" * 60)
    print("📈 SUMMARY")
    print("=" * 60)
    
    # Print summary
    for column, data in stats.items():
        basic = data['basic_stats']
        print(f"{column}: {data['type']} | "
              f"{basic['count']['valid']}/{basic['count']['total']} valid | "
              f"{basic['uniqueness']['unique_count']} unique")


if __name__ == "__main__":
    main()