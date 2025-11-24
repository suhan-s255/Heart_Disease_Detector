"""
Tests for synthetic data generation functionality
"""
import unittest
import sys
import os
from pathlib import Path
import pandas as pd

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSyntheticGeneration(unittest.TestCase):
    """Test synthetic data generation and compatibility"""
    
    def test_cleaned_dataset_exists(self):
        """Test that the cleaned dataset exists"""
        dataset_path = project_root / 'data' / 'cleaned_merged_heart_dataset.csv'
        self.assertTrue(dataset_path.exists(), "Cleaned dataset file should exist")
    
    def test_synthetic_output_exists(self):
        """Test that synthetic augmented dataset exists (after generation)"""
        output_path = project_root / 'data' / 'synthetic_augmented_heart_dataset.csv'
        
        # This test only runs if the file exists (generation script was run)
        if output_path.exists():
            self.assertTrue(output_path.exists(), "Synthetic dataset should exist after generation")
            
            # Load and validate
            df = pd.read_csv(output_path)
            
            # Test row count
            self.assertEqual(
                len(df), 8768, 
                "Synthetic dataset should have exactly 8,768 rows"
            )
            
            # Test column count
            self.assertEqual(
                len(df.columns), 14,
                "Synthetic dataset should have 14 columns"
            )
        else:
            self.skipTest("Synthetic dataset not yet generated - run generate_synthetic_compatible.py first")
    
    def test_synthetic_dataset_compatibility(self):
        """Test that synthetic dataset has compatible structure with cleaned dataset"""
        cleaned_path = project_root / 'data' / 'cleaned_merged_heart_dataset.csv'
        synthetic_path = project_root / 'data' / 'synthetic_augmented_heart_dataset.csv'
        
        if not synthetic_path.exists():
            self.skipTest("Synthetic dataset not yet generated - run generate_synthetic_compatible.py first")
        
        cleaned_df = pd.read_csv(cleaned_path)
        synthetic_df = pd.read_csv(synthetic_path)
        
        # Test column names match
        self.assertEqual(
            list(cleaned_df.columns), 
            list(synthetic_df.columns),
            "Column names should match exactly"
        )
        
        # Test column order matches
        for i, col in enumerate(cleaned_df.columns):
            self.assertEqual(
                cleaned_df.columns[i],
                synthetic_df.columns[i],
                f"Column order mismatch at position {i}"
            )
        
        # Test data types match
        for col in cleaned_df.columns:
            self.assertEqual(
                cleaned_df[col].dtype,
                synthetic_df[col].dtype,
                f"Data type mismatch for column {col}"
            )
    
    def test_synthetic_dataset_has_target_column(self):
        """Test that synthetic dataset has target column"""
        synthetic_path = project_root / 'data' / 'synthetic_augmented_heart_dataset.csv'
        
        if not synthetic_path.exists():
            self.skipTest("Synthetic dataset not yet generated - run generate_synthetic_compatible.py first")
        
        df = pd.read_csv(synthetic_path)
        
        # Check for target column
        self.assertIn('target', df.columns, "Dataset should have 'target' column")
        
        # Check target values are 0 or 1
        unique_targets = df['target'].unique()
        self.assertTrue(
            all(val in [0, 1] for val in unique_targets),
            "Target column should only contain 0 or 1"
        )
    
    def test_synthetic_dataset_no_missing_values(self):
        """Test that synthetic dataset has no missing values"""
        synthetic_path = project_root / 'data' / 'synthetic_augmented_heart_dataset.csv'
        
        if not synthetic_path.exists():
            self.skipTest("Synthetic dataset not yet generated - run generate_synthetic_compatible.py first")
        
        df = pd.read_csv(synthetic_path)
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        self.assertEqual(
            missing_counts.sum(),
            0,
            "Synthetic dataset should have no missing values"
        )
    
    def test_evaluation_reports_exist(self):
        """Test that evaluation reports exist (after evaluation)"""
        json_report = project_root / 'reports' / 'synthetic_evaluation.json'
        text_report = project_root / 'reports' / 'synthetic_evaluation.txt'
        
        # These tests only run if the files exist (evaluation script was run)
        if json_report.exists():
            self.assertTrue(json_report.exists(), "JSON report should exist")
            
            # Verify it's valid JSON
            import json
            with open(json_report, 'r') as f:
                data = json.load(f)
                self.assertIn('ml_evaluation', data, "Report should have ML evaluation")
        else:
            self.skipTest("Evaluation not yet run - run evaluate_synthetic_and_model.py first")
        
        if text_report.exists():
            self.assertTrue(text_report.exists(), "Text report should exist")
            
            # Verify it's not empty
            with open(text_report, 'r') as f:
                content = f.read()
                self.assertGreater(len(content), 0, "Text report should not be empty")


class TestGenerationScript(unittest.TestCase):
    """Test the generation script can be imported"""
    
    def test_generation_script_imports(self):
        """Test that generation script can be imported"""
        try:
            sys.path.insert(0, str(project_root / 'scripts'))
            import generate_synthetic_compatible
            self.assertTrue(True, "Generation script should import successfully")
        except ImportError as e:
            self.fail(f"Failed to import generation script: {str(e)}")
    
    def test_evaluation_script_imports(self):
        """Test that evaluation script can be imported"""
        try:
            sys.path.insert(0, str(project_root / 'scripts'))
            import evaluate_synthetic_and_model
            self.assertTrue(True, "Evaluation script should import successfully")
        except ImportError as e:
            self.fail(f"Failed to import evaluation script: {str(e)}")


if __name__ == '__main__':
    unittest.main()
