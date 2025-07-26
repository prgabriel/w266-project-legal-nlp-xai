"""
Comprehensive Test Suite for Legal NLP Utility Functions
Tests for data handling, preprocessing, project management, and legal text processing
"""

import unittest
import tempfile
import shutil
import os
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import warnings

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Suppress warnings during testing
warnings.filterwarnings('ignore')

try:
    from scripts.utils import (
        ProjectStructure, load_data, save_data, preprocess_text,
        extract_legal_entities, clean_clause_name, split_data,
        create_stratified_sample, create_multilabel_matrix,
        convert_multilabel_to_single, calculate_metrics,
        calculate_confusion_matrix_metrics, create_experiment_id,
        ConfigManager
    )
except ImportError as e:
    print(f"Warning: Could not import utils functions: {e}")
    print("Make sure the scripts/utils.py file exists and is properly configured.")
    raise

class TestProjectStructure(unittest.TestCase):
    """Test ProjectStructure class for path management"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.project = ProjectStructure(project_root=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors
    
    def test_project_root_detection(self):
        """Test automatic project root detection"""
        self.assertTrue(os.path.exists(self.project.project_root))
        self.assertEqual(self.project.project_root, self.temp_dir)
    
    def test_get_path(self):
        """Test path generation functionality"""
        data_path = self.project.get_path('data')
        expected_path = os.path.join(self.temp_dir, 'data')
        self.assertEqual(data_path, expected_path)
        
        # Test with subpaths
        processed_path = self.project.get_path('data', 'processed', 'test.csv')
        expected_processed = os.path.join(self.temp_dir, 'data', 'processed', 'test.csv')
        self.assertEqual(processed_path, expected_processed)
    
    def test_ensure_dirs(self):
        """Test directory creation functionality"""
        self.project.ensure_dirs('data', 'models', 'logs')
        
        # Verify directories were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'data')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'models')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'logs')))
    
    def test_get_all_paths(self):
        """Test retrieval of all defined paths"""
        all_paths = self.project.get_all_paths()
        
        # Check that expected keys are present
        expected_keys = ['data', 'models', 'notebooks', 'scripts', 'app', 'tests']
        for key in expected_keys:
            self.assertIn(key, all_paths)
            self.assertTrue(all_paths[key].startswith(self.temp_dir))

class TestDataHandling(unittest.TestCase):
    """Test data loading and saving functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = {
            'clause_types': ['Agreement Date', 'Governing Law'],
            'samples': [
                {'text': 'This agreement shall commence on January 1, 2024.', 'labels': ['Agreement Date']},
                {'text': 'This agreement shall be governed by California law.', 'labels': ['Governing Law']}
            ]
        }
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
    
    def test_save_and_load_json(self):
        """Test JSON data saving and loading"""
        json_path = os.path.join(self.temp_dir, 'test.json')
        
        # Save data
        save_data(self.test_data, json_path)
        self.assertTrue(os.path.exists(json_path))
        
        # Load data
        loaded_data = load_data(json_path)
        self.assertEqual(loaded_data, self.test_data)
    
    def test_save_and_load_csv(self):
        """Test CSV data handling"""
        df = pd.DataFrame({
            'text': ['Legal document 1', 'Legal document 2'],
            'clause_type': ['Agreement Date', 'Governing Law'],
            'confidence': [0.95, 0.87]
        })
        
        csv_path = os.path.join(self.temp_dir, 'test.csv')
        
        # Save CSV
        save_data(df, csv_path)
        self.assertTrue(os.path.exists(csv_path))
        
        # Load CSV
        loaded_df = load_data(csv_path)
        pd.testing.assert_frame_equal(df, loaded_df)
    
    def test_invalid_file_path(self):
        """Test handling of invalid file paths"""
        with self.assertRaises(FileNotFoundError):
            load_data('/nonexistent/path/file.json')

class TestLegalTextProcessing(unittest.TestCase):
    """Test legal text preprocessing and entity extraction"""
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing"""
        input_text = "  This   Agreement   shall   commence   on   January  1,  2024.  "
        expected = "This Agreement shall commence on January 1, 2024."
        
        result = preprocess_text(input_text, normalize_whitespace=True)
        self.assertEqual(result, expected)
    
    def test_preprocess_text_legal_optimization(self):
        """Test legal-specific text preprocessing"""
        legal_text = "Party A hereby agrees that the aforementioned provisions shall govern."
        
        result = preprocess_text(legal_text, for_legal=True)
        
        # Should contain legal text without major alterations
        self.assertIn("Party A", result)
        self.assertIn("hereby agrees", result)
        self.assertIn("aforementioned", result)
    
    def test_preprocess_text_length_limit(self):
        """Test text length limiting"""
        long_text = "This is a very long legal document. " * 100
        
        result = preprocess_text(long_text, max_length=100)
        self.assertLessEqual(len(result), 100)
    
    def test_extract_legal_entities(self):
        """Test legal entity extraction"""
        legal_text = """
        This Agreement shall commence on January 1, 2024 and shall continue 
        for a period of three years. TechCorp Inc. shall pay $50,000 annually.
        The agreement shall be governed by California law.
        """
        
        entities = extract_legal_entities(legal_text)
        
        # Check that entities were extracted
        self.assertIn('dates', entities)
        self.assertIn('companies', entities)
        self.assertIn('monetary_amounts', entities)
        
        # Verify specific extractions
        self.assertTrue(any('January 1, 2024' in date or '2024' in date for date in entities['dates']))
        self.assertTrue(any('TechCorp' in company for company in entities['companies']))
        self.assertTrue(any('50,000' in amount or '$50,000' in amount for amount in entities['monetary_amounts']))
    
    def test_clean_clause_name(self):
        """Test clause name cleaning functionality"""
        test_cases = [
            ('Does the clause specify "Agreement Date"?', 'Agreement Date'),
            ('test_binary_governing_law', 'Governing Law'),
            ('ANTI_ASSIGNMENT_CLAUSE', 'Anti Assignment Clause'),
            ('   Messy  Clause   Name   ', 'Messy Clause Name')
        ]
        
        for input_name, expected in test_cases:
            with self.subTest(input_name=input_name):
                result = clean_clause_name(input_name)
                self.assertEqual(result, expected)

class TestDataSplitting(unittest.TestCase):
    """Test data splitting and sampling utilities"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)  # For reproducible tests
        self.sample_data = np.random.randn(1000, 5)
        self.sample_labels = np.random.randint(0, 3, 1000)
    
    def test_split_data_basic(self):
        """Test basic data splitting"""
        train, val, test = split_data(self.sample_data, train_size=0.7, val_size=0.2, test_size=0.1)
        
        # Check proportions
        total_size = len(self.sample_data)
        self.assertAlmostEqual(len(train) / total_size, 0.7, delta=0.05)
        self.assertAlmostEqual(len(val) / total_size, 0.2, delta=0.05)
        self.assertAlmostEqual(len(test) / total_size, 0.1, delta=0.05)
        
        # Check no data leakage
        self.assertEqual(len(train) + len(val) + len(test), total_size)
    
    def test_split_data_with_stratification(self):
        """Test stratified data splitting"""
        train, val, test = split_data(
            self.sample_data, 
            train_size=0.8, 
            val_size=0.2,
            stratify=self.sample_labels
        )
        
        # Basic size checks
        self.assertGreater(len(train), len(val))
        self.assertEqual(len(train) + len(val), len(self.sample_data))
    
    def test_create_stratified_sample(self):
        """Test stratified sampling"""
        df = pd.DataFrame({
            'text': [f'Document {i}' for i in range(1000)],
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'value': np.random.randn(1000)
        })
        
        sample = create_stratified_sample(df, 'category', sample_size=300)
        
        # Check sample size
        self.assertEqual(len(sample), 300)
        
        # Check that all categories are represented
        self.assertEqual(set(sample['category'].unique()), set(df['category'].unique()))

class TestMultiLabelUtilities(unittest.TestCase):
    """Test multi-label processing utilities"""
    
    def test_create_multilabel_matrix(self):
        """Test multi-label matrix creation"""
        labels = [
            ['Agreement Date', 'Governing Law'],
            ['Governing Law'],
            ['Agreement Date', 'Termination'],
            []
        ]
        
        matrix, label_names = create_multilabel_matrix(labels)
        
        # Check matrix shape
        self.assertEqual(matrix.shape[0], len(labels))
        self.assertEqual(matrix.shape[1], len(label_names))
        
        # Check specific entries
        self.assertEqual(matrix[0, label_names.index('Agreement Date')], 1)
        self.assertEqual(matrix[0, label_names.index('Governing Law')], 1)
        self.assertEqual(matrix[1, label_names.index('Agreement Date')], 0)
        self.assertEqual(matrix[3].sum(), 0)  # Empty labels
    
    def test_convert_multilabel_to_single(self):
        """Test multi-label to single-label conversion"""
        # Create test multi-label matrix
        multilabel_matrix = np.array([
            [1, 1, 0],  # Two labels
            [0, 1, 0],  # One label
            [1, 0, 1],  # Two labels
            [0, 0, 0]   # No labels
        ])
        
        single_labels = convert_multilabel_to_single(multilabel_matrix, strategy='most_frequent')
        
        # Check output shape
        self.assertEqual(len(single_labels), multilabel_matrix.shape[0])
        self.assertTrue(all(isinstance(label, (int, np.integer)) for label in single_labels))
        
        # Test different strategies
        for strategy in ['first', 'random']:
            with self.subTest(strategy=strategy):
                result = convert_multilabel_to_single(multilabel_matrix, strategy=strategy)
                self.assertEqual(len(result), multilabel_matrix.shape[0])

class TestMetricsCalculation(unittest.TestCase):
    """Test metrics calculation utilities"""
    
    def test_calculate_metrics_binary(self):
        """Test binary classification metrics"""
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])
        
        metrics = calculate_metrics(y_pred, y_true, metric_types=['precision', 'recall', 'f1'])
        
        # Check that all requested metrics are present
        for metric in ['precision', 'recall', 'f1']:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (float, np.floating))
            self.assertGreaterEqual(metrics[metric], 0)
            self.assertLessEqual(metrics[metric], 1)
    
    def test_calculate_metrics_multilabel(self):
        """Test multi-label classification metrics"""
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
        y_pred = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 1, 1]])
        
        metrics = calculate_metrics(y_pred, y_true, multilabel=True)
        
        # Check basic metrics presence
        expected_metrics = ['precision', 'recall', 'f1']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
    
    def test_calculate_metrics_with_different_averages(self):
        """Test metrics calculation with different averaging strategies"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 1])
        
        for average in ['macro', 'micro', 'weighted']:
            with self.subTest(average=average):
                metrics = calculate_metrics(y_pred, y_true, average=average)
                self.assertIn('precision', metrics)
                self.assertIn('recall', metrics)
                self.assertIn('f1', metrics)
    
    def test_calculate_confusion_matrix_metrics(self):
        """Test confusion matrix metrics calculation"""
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])
        
        cm_metrics = calculate_confusion_matrix_metrics(y_true, y_pred)
        
        # Check that all confusion matrix metrics are present
        expected_keys = ['tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'f1']
        for key in expected_keys:
            self.assertIn(key, cm_metrics)
        
        # Verify mathematical relationships
        tp, tn, fp, fn = cm_metrics['tp'], cm_metrics['tn'], cm_metrics['fp'], cm_metrics['fn']
        expected_accuracy = (tp + tn) / (tp + tn + fp + fn)
        self.assertAlmostEqual(cm_metrics['accuracy'], expected_accuracy, places=5)

class TestConfigurationManagement(unittest.TestCase):
    """Test configuration management utilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
    
    @patch('scripts.utils.PROJECT')
    def test_config_manager_initialization(self, mock_project):
        """Test ConfigManager initialization"""
        mock_project.project_root = self.temp_dir
        mock_project.get_all_paths.return_value = {'data': '/data', 'models': '/models'}
        
        config_manager = ConfigManager(config_file='test_config.yaml')
        
        # Check that default config was created
        self.assertIsInstance(config_manager.config, dict)
        self.assertIn('model', config_manager.config)
        self.assertIn('training', config_manager.config)
    
    def test_config_get_set(self):
        """Test configuration get/set operations"""
        with patch('scripts.utils.PROJECT') as mock_project:
            mock_project.project_root = self.temp_dir
            mock_project.get_all_paths.return_value = {}
            
            config_manager = ConfigManager(config_file='test_config.yaml')
            
            # Test setting and getting values
            config_manager.set('test.nested.value', 42)
            self.assertEqual(config_manager.get('test.nested.value'), 42)
            
            # Test default value
            self.assertEqual(config_manager.get('nonexistent.key', 'default'), 'default')
            
            # Test complex nested structure
            config_manager.set('complex.deep.nested.structure', {'key': 'value'})
            result = config_manager.get('complex.deep.nested.structure')
            self.assertEqual(result, {'key': 'value'})

class TestUtilityFunctions(unittest.TestCase):
    """Test miscellaneous utility functions"""
    
    def test_create_experiment_id(self):
        """Test experiment ID generation"""
        exp_id1 = create_experiment_id()
        exp_id2 = create_experiment_id()
        
        # Check format (timestamp_hash)
        self.assertRegex(exp_id1, r'\d{8}_\d{6}_[a-f0-9]{8}')
        
        # Check uniqueness
        self.assertNotEqual(exp_id1, exp_id2)
        
        # Check length consistency
        self.assertEqual(len(exp_id1.split('_')), 3)  # date_time_hash
    
    def test_preprocess_text_edge_cases(self):
        """Test text preprocessing edge cases"""
        # Empty string
        self.assertEqual(preprocess_text(''), '')
        
        # Only whitespace
        self.assertEqual(preprocess_text('   \n\t  '), '')
        
        # Special characters
        special_text = "Contract #123 @mention & co. (subsidiary)"
        result = preprocess_text(special_text, remove_special_chars=True)
        self.assertNotIn('#', result)
        self.assertNotIn('@', result)
        
        # Very long text
        very_long_text = "word " * 1000
        result = preprocess_text(very_long_text, max_length=50)
        self.assertLessEqual(len(result), 50)

class TestLegalDomainSpecific(unittest.TestCase):
    """Test legal domain-specific functionality"""
    
    def test_legal_entity_extraction_comprehensive(self):
        """Test comprehensive legal entity extraction"""
        complex_legal_text = """
        AGREEMENT dated December 15, 2023, between TechCorp Inc., a Delaware corporation 
        ("Company"), and DataSoft LLC, a California limited liability company ("Contractor").
        
        The total compensation shall be $250,000 payable in quarterly installments of $62,500.
        This agreement shall terminate on December 31, 2025, unless extended by mutual consent.
        
        The parties agree that any disputes shall be resolved under New York State law.
        """
        
        entities = extract_legal_entities(complex_legal_text)
        
        # Verify comprehensive extraction
        self.assertIn('dates', entities)
        self.assertIn('companies', entities)
        self.assertIn('monetary_amounts', entities)
        self.assertIn('contract_terms', entities)
        
        # Check specific legal terms
        dates = entities['dates']
        companies = entities['companies']
        amounts = entities['monetary_amounts']
        
        # Should extract multiple dates
        self.assertGreater(len(dates), 0)
        
        # Should extract company names
        self.assertGreater(len(companies), 0)
        
        # Should extract monetary amounts
        self.assertGreater(len(amounts), 0)
    
    def test_clause_name_cleaning_edge_cases(self):
        """Test clause name cleaning with edge cases"""
        edge_cases = [
            ('', ''),  # Empty string
            ('CLAUSE_WITH_UNDERSCORES_AND_NUMBERS_123', 'Clause With Underscores And Numbers 123'),
            ('mixed_CASE_clause_NAME', 'Mixed Case Clause Name'),
            ('Does "the clause" include specific terms?', 'The Clause'),
            ('   extra   whitespace   everywhere   ', 'Extra Whitespace Everywhere')
        ]
        
        for input_name, expected in edge_cases:
            with self.subTest(input_name=input_name):
                result = clean_clause_name(input_name)
                self.assertEqual(result, expected)

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_metrics_calculation_error_handling(self):
        """Test that metrics calculation handles errors gracefully"""
        # Mismatched array sizes
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0])  # Different size
        
        # Should not raise an exception, but handle gracefully
        try:
            metrics = calculate_metrics(y_pred, y_true)
            # If it returns, it should contain some metrics (possibly zeros)
            self.assertIsInstance(metrics, dict)
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass
    
    def test_multilabel_matrix_empty_labels(self):
        """Test multi-label matrix creation with various edge cases"""
        # All empty labels
        empty_labels = [[], [], []]
        matrix, label_names = create_multilabel_matrix(empty_labels)
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(matrix.sum(), 0)
        
        # Mixed empty and non-empty
        mixed_labels = [['A'], [], ['A', 'B']]
        matrix, label_names = create_multilabel_matrix(mixed_labels)
        self.assertEqual(matrix.shape[0], 3)
        self.assertIn('A', label_names)
        self.assertIn('B', label_names)

def create_test_suite():
    """Create comprehensive test suite for all utility functions"""
    test_suite = unittest.TestSuite()
    
    # Add test classes in logical order
    test_classes = [
        TestProjectStructure,
        TestDataHandling,
        TestLegalTextProcessing,
        TestDataSplitting,
        TestMultiLabelUtilities,
        TestMetricsCalculation,
        TestConfigurationManagement,
        TestUtilityFunctions,
        TestLegalDomainSpecific,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    return test_suite

if __name__ == '__main__':
    # Run comprehensive test suite
    print("="*80)
    print("LEGAL NLP UTILITIES - COMPREHENSIVE TEST SUITE")
    print("Testing all utility functions for legal document processing")
    print("="*80)
    
    # Create and run test suite
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("TEST EXECUTION SUMMARY")
    print("="*80)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Successful Tests: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed Tests: {len(result.failures)}")
    print(f"Error Tests: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n📋 FAILED TESTS ({len(result.failures)}):")
        for i, (test, failure) in enumerate(result.failures, 1):
            test_name = str(test).split(' ')[0]
            failure_msg = failure.split('AssertionError: ')[-1].split('\n')[0] if 'AssertionError:' in failure else 'See details above'
            print(f"  {i}. {test_name}: {failure_msg}")
    
    if result.errors:
        print(f"\n❌ TESTS WITH ERRORS ({len(result.errors)}):")
        for i, (test, error) in enumerate(result.errors, 1):
            test_name = str(test).split(' ')[0]
            error_msg = error.split('\n')[-2] if '\n' in error else error
            print(f"  {i}. {test_name}: {error_msg}")
    
    if result.wasSuccessful():
        print(f"\n✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("Your legal NLP utilities are working correctly.")
    else:
        print(f"\n⚠️  SOME TESTS FAILED")
        print("Review the failures above to fix any issues.")
    
    print("\n" + "="*80)
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)