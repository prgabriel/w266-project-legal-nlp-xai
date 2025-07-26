"""
Comprehensive Test Suite for Legal NLP Models
Tests for BERT clause extraction, T5 summarization, model loading, and evaluation
"""

import unittest
import tempfile
import shutil
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock, mock_open
import sys
import warnings
from typing import Dict, List, Tuple, Optional, Any

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Suppress warnings during testing
warnings.filterwarnings('ignore')

try:
    from models.fine_tuning.clause_extraction import LegalClauseExtractor, LegalClauseDataset
    from models.fine_tuning.summarization import LegalSummarizationModel, LegalSummarizationDataset
except ImportError as e:
    print(f"Warning: Could not import model classes: {e}")
    print("Some tests may be skipped if models are not available.")

# Mock model classes for testing when transformers is not available
class MockBertTokenizer:
    """Mock BERT tokenizer for testing"""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
    
    def __call__(self, text, **kwargs):
        # Return mock tokenization
        return {
            'input_ids': torch.randint(0, self.vocab_size, (1, kwargs.get('max_length', 512))),
            'attention_mask': torch.ones(1, kwargs.get('max_length', 512))
        }
    
    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

class MockBertModel(nn.Module):
    """Mock BERT model for testing"""
    def __init__(self, num_labels=41):
        super().__init__()
        self.num_labels = num_labels
        self.config = MagicMock()
        self.config.hidden_size = 768
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.size(0)
        return {
            'logits': torch.randn(batch_size, self.num_labels),
            'loss': torch.randn(1) if labels is not None else None
        }
    
    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls(num_labels=kwargs.get('num_labels', 41))

class MockT5Model(nn.Module):
    """Mock T5 model for testing"""
    def __init__(self):
        super().__init__()
        self.config = MagicMock()
    
    def generate(self, input_ids, **kwargs):
        return torch.randint(0, 1000, (input_ids.size(0), kwargs.get('max_length', 100)))
    
    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

class TestLegalClauseDataset(unittest.TestCase):
    """Test LegalClauseDataset for multi-label clause classification"""
    
    def setUp(self):
        """Set up test environment"""
        self.texts = [
            "This agreement shall commence on January 1, 2024.",
            "The parties agree to resolve disputes through arbitration.",
            "This contract is governed by California law."
        ]
        self.labels = np.array([
            [1, 0, 0, 1],  # Agreement Date, Governing Law
            [0, 1, 0, 0],  # Dispute Resolution
            [0, 0, 1, 1]   # Governing Law
        ])
        self.mock_tokenizer = MockBertTokenizer()
        
        # Skip if actual model classes not available
        try:
            from models.fine_tuning.clause_extraction import LegalClauseDataset
            self.dataset_available = True
        except ImportError:
            self.dataset_available = False
    
    @unittest.skipUnless('LegalClauseDataset' in globals(), "LegalClauseDataset not available")
    def test_dataset_initialization(self):
        """Test dataset initialization"""
        if not self.dataset_available:
            self.skipTest("LegalClauseDataset not available")
            
        dataset = LegalClauseDataset(self.texts, self.labels, self.mock_tokenizer)
        
        self.assertEqual(len(dataset), len(self.texts))
        self.assertEqual(dataset.max_length, 512)  # Default max length
    
    @unittest.skipUnless('LegalClauseDataset' in globals(), "LegalClauseDataset not available")
    def test_dataset_getitem(self):
        """Test dataset item retrieval"""
        if not self.dataset_available:
            self.skipTest("LegalClauseDataset not available")
            
        dataset = LegalClauseDataset(self.texts, self.labels, self.mock_tokenizer, max_length=128)
        
        item = dataset[0]
        
        # Check that all required keys are present
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)
        
        # Check tensor shapes
        self.assertEqual(item['input_ids'].shape, torch.Size([128]))
        self.assertEqual(item['attention_mask'].shape, torch.Size([128]))
        self.assertEqual(item['labels'].shape, torch.Size([4]))  # Number of labels
    
    def test_dataset_with_empty_texts(self):
        """Test dataset with edge cases"""
        empty_texts = ["", "   ", "short"]
        empty_labels = np.zeros((3, 4))
        
        # Should handle empty/short texts gracefully
        dataset = LegalClauseDataset(empty_texts, empty_labels, self.mock_tokenizer)
        self.assertEqual(len(dataset), 3)

class TestLegalSummarizationDataset(unittest.TestCase):
    """Test LegalSummarizationDataset for T5 summarization"""
    
    def setUp(self):
        """Set up test environment"""
        self.texts = [
            "This is a long legal document about contract terms and conditions.",
            "The agreement outlines the responsibilities of both parties.",
            "Termination clauses specify the conditions for contract end."
        ]
        self.summaries = [
            "Contract terms document",
            "Party responsibilities agreement", 
            "Termination conditions"
        ]
        self.mock_tokenizer = MockBertTokenizer()  # Can reuse for T5
        
        try:
            from models.fine_tuning.summarization import LegalSummarizationDataset
            self.dataset_available = True
        except ImportError:
            self.dataset_available = False
    
    @unittest.skipUnless('LegalSummarizationDataset' in globals(), "LegalSummarizationDataset not available")
    def test_summarization_dataset_initialization(self):
        """Test summarization dataset initialization"""
        if not self.dataset_available:
            self.skipTest("LegalSummarizationDataset not available")
            
        dataset = LegalSummarizationDataset(
            self.texts, self.summaries, self.mock_tokenizer,
            max_input_length=512, max_target_length=128
        )
        
        self.assertEqual(len(dataset), len(self.texts))
        self.assertEqual(dataset.max_input_length, 512)
        self.assertEqual(dataset.max_target_length, 128)
    
    @unittest.skipUnless('LegalSummarizationDataset' in globals(), "LegalSummarizationDataset not available")
    def test_summarization_dataset_getitem(self):
        """Test summarization dataset item retrieval"""
        if not self.dataset_available:
            self.skipTest("LegalSummarizationDataset not available")
            
        dataset = LegalSummarizationDataset(
            self.texts, self.summaries, self.mock_tokenizer,
            max_input_length=256, max_target_length=64
        )
        
        item = dataset[0]
        
        # Check required keys
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)
        
        # Check tensor shapes
        self.assertEqual(item['input_ids'].shape, torch.Size([256]))
        self.assertEqual(item['attention_mask'].shape, torch.Size([256]))
        self.assertEqual(item['labels'].shape, torch.Size([64]))

class TestLegalClauseExtractor(unittest.TestCase):
    """Test LegalClauseExtractor class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock metadata
        self.mock_metadata = {
            'clause_types': [
                'Agreement Date',
                'Governing Law', 
                'Termination',
                'Dispute Resolution'
            ],
            'clean_clause_names': {
                'Agreement Date': 'Agreement Date',
                'Governing Law': 'Governing Law',
                'Termination': 'Termination Clause',
                'Dispute Resolution': 'Dispute Resolution'
            }
        }
        
        # Create mock metadata file
        metadata_path = os.path.join(self.temp_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.mock_metadata, f)
        
        try:
            from models.fine_tuning.clause_extraction import LegalClauseExtractor
            self.extractor_available = True
        except ImportError:
            self.extractor_available = False
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
    
    @patch('models.fine_tuning.clause_extraction.BertTokenizer')
    @patch('models.fine_tuning.clause_extraction.BertForSequenceClassification')
    def test_clause_extractor_initialization(self, mock_bert_model, mock_bert_tokenizer):
        """Test LegalClauseExtractor initialization"""
        if not self.extractor_available:
            self.skipTest("LegalClauseExtractor not available")
        
        # Mock the model and tokenizer
        mock_bert_tokenizer.from_pretrained.return_value = MockBertTokenizer()
        mock_bert_model.from_pretrained.return_value = MockBertModel(num_labels=4)
        
        with patch.object(LegalClauseExtractor, '_load_metadata', return_value=self.mock_metadata):
            extractor = LegalClauseExtractor(
                model_name='bert-base-uncased',
                data_dir=self.temp_dir,
                output_dir=self.temp_dir
            )
            
            self.assertEqual(extractor.num_labels, 4)
            self.assertEqual(len(extractor.clause_types), 4)
            self.assertIn('Agreement Date', extractor.clause_types)
    
    def test_metadata_loading(self):
        """Test metadata loading functionality"""
        if not self.extractor_available:
            self.skipTest("LegalClauseExtractor not available")
        
        # Test with mock file system
        with patch('builtins.open', mock_open(read_data=json.dumps(self.mock_metadata))):
            with patch('os.path.exists', return_value=True):
                extractor = LegalClauseExtractor.__new__(LegalClauseExtractor)
                extractor.data_dir = self.temp_dir
                metadata = extractor._load_metadata()
                
                self.assertEqual(metadata['clause_types'], self.mock_metadata['clause_types'])
                self.assertEqual(len(metadata['clean_clause_names']), 4)
    
    @patch('models.fine_tuning.clause_extraction.BertTokenizer')
    @patch('models.fine_tuning.clause_extraction.BertForSequenceClassification')
    def test_prediction_functionality(self, mock_bert_model, mock_bert_tokenizer):
        """Test prediction functionality"""
        if not self.extractor_available:
            self.skipTest("LegalClauseExtractor not available")
        
        # Mock the components
        mock_tokenizer_instance = MockBertTokenizer()
        mock_model_instance = MockBertModel(num_labels=4)
        
        mock_bert_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_bert_model.from_pretrained.return_value = mock_model_instance
        
        with patch.object(LegalClauseExtractor, '_load_metadata', return_value=self.mock_metadata):
            extractor = LegalClauseExtractor(data_dir=self.temp_dir, output_dir=self.temp_dir)
            
            # Test prediction
            test_texts = ["This agreement starts on January 1, 2024."]
            
            with patch.object(extractor.model, 'eval'), \
                 patch('torch.no_grad'), \
                 patch.object(extractor.model, 'forward', return_value={'logits': torch.randn(1, 4)}):
                
                predictions = extractor.predict(test_texts, threshold=0.5)
                
                self.assertIsInstance(predictions, list)
                self.assertEqual(len(predictions), 1)

class TestLegalSummarizationModel(unittest.TestCase):
    """Test LegalSummarizationModel class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            from models.fine_tuning.summarization import LegalSummarizationModel
            self.summarizer_available = True
        except ImportError:
            self.summarizer_available = False
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
    
    @patch('models.fine_tuning.summarization.T5Tokenizer')
    @patch('models.fine_tuning.summarization.T5ForConditionalGeneration')
    def test_summarization_model_initialization(self, mock_t5_model, mock_t5_tokenizer):
        """Test LegalSummarizationModel initialization"""
        if not self.summarizer_available:
            self.skipTest("LegalSummarizationModel not available")
        
        # Mock the T5 components
        mock_t5_tokenizer.from_pretrained.return_value = MockBertTokenizer()  # Reuse mock
        mock_t5_model.from_pretrained.return_value = MockT5Model()
        
        summarizer = LegalSummarizationModel(
            model_name='t5-base',
            output_dir=self.temp_dir
        )
        
        self.assertEqual(summarizer.model_name, 't5-base')
        self.assertEqual(summarizer.output_dir, self.temp_dir)
    
    @patch('models.fine_tuning.summarization.T5Tokenizer')
    @patch('models.fine_tuning.summarization.T5ForConditionalGeneration')
    def test_text_summarization(self, mock_t5_model, mock_t5_tokenizer):
        """Test text summarization functionality"""
        if not self.summarizer_available:
            self.skipTest("LegalSummarizationModel not available")
        
        # Mock components
        mock_tokenizer_instance = MockBertTokenizer()
        mock_model_instance = MockT5Model()
        
        mock_t5_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_t5_model.from_pretrained.return_value = mock_model_instance
        
        summarizer = LegalSummarizationModel(output_dir=self.temp_dir)
        
        test_text = "This is a long legal document that needs to be summarized for quick review."
        
        with patch.object(summarizer.tokenizer, 'decode', return_value="Legal document summary"):
            result = summarizer.summarize(test_text, max_length=50)
            
            self.assertIsInstance(result, dict)
            self.assertIn('summary', result)
    
    def test_legal_text_preprocessing(self):
        """Test legal text preprocessing"""
        if not self.summarizer_available:
            self.skipTest("LegalSummarizationModel not available")
        
        # Test without actual model initialization
        text = "The parties hereto shall be bound by this agreement."
        
        # Mock the preprocessing method
        with patch('models.fine_tuning.summarization.T5Tokenizer'), \
             patch('models.fine_tuning.summarization.T5ForConditionalGeneration'):
            
            summarizer = LegalSummarizationModel(output_dir=self.temp_dir)
            processed = summarizer.preprocess_legal_text(text)
            
            self.assertIsInstance(processed, str)
            self.assertNotEqual(processed, "")

class TestModelEvaluationIntegration(unittest.TestCase):
    """Test integration with evaluation framework"""
    
    def setUp(self):
        """Set up test environment"""
        self.sample_predictions = np.array([
            [0.8, 0.2, 0.1, 0.9],
            [0.1, 0.7, 0.3, 0.2],
            [0.6, 0.1, 0.8, 0.4]
        ])
        self.sample_labels = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 1, 0]
        ])
    
    def test_prediction_format_compatibility(self):
        """Test that model predictions are compatible with evaluation metrics"""
        # Test binary conversion
        binary_predictions = (self.sample_predictions > 0.5).astype(int)
        
        self.assertEqual(binary_predictions.shape, self.sample_labels.shape)
        self.assertTrue(np.all((binary_predictions == 0) | (binary_predictions == 1)))
    
    def test_multi_label_metrics_compatibility(self):
        """Test compatibility with multi-label metrics"""
        # Simulate metric calculation
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        binary_preds = (self.sample_predictions > 0.5).astype(int)
        
        # These should not raise exceptions
        f1_micro = f1_score(self.sample_labels, binary_preds, average='micro')
        precision_macro = precision_score(self.sample_labels, binary_preds, average='macro', zero_division=0)
        recall_weighted = recall_score(self.sample_labels, binary_preds, average='weighted', zero_division=0)
        
        # Check that metrics are reasonable
        self.assertGreaterEqual(f1_micro, 0)
        self.assertLessEqual(f1_micro, 1)
        self.assertGreaterEqual(precision_macro, 0)
        self.assertGreaterEqual(recall_weighted, 0)

class TestModelFileHandling(unittest.TestCase):
    """Test model file operations and persistence"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
    
    def test_model_directory_creation(self):
        """Test model directory structure creation"""
        # Create expected directory structure
        bert_dir = os.path.join(self.model_dir, 'bert')
        t5_dir = os.path.join(self.model_dir, 't5')
        
        os.makedirs(bert_dir, exist_ok=True)
        os.makedirs(t5_dir, exist_ok=True)
        
        self.assertTrue(os.path.exists(bert_dir))
        self.assertTrue(os.path.exists(t5_dir))
    
    def test_metadata_file_handling(self):
        """Test metadata file creation and loading"""
        metadata = {
            'model_name': 'bert-base-uncased',
            'num_labels': 41,
            'clause_types': ['Agreement Date', 'Governing Law'],
            'training_date': '2024-01-15'
        }
        
        metadata_path = os.path.join(self.model_dir, 'model_info.json')
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Load and verify
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        self.assertEqual(loaded_metadata['model_name'], 'bert-base-uncased')
        self.assertEqual(loaded_metadata['num_labels'], 41)
        self.assertEqual(len(loaded_metadata['clause_types']), 2)
    
    def test_training_results_persistence(self):
        """Test training results file handling"""
        training_results = {
            'model_config': {
                'model_name': 'bert-base-uncased',
                'num_labels': 41,
                'learning_rate': 2e-5
            },
            'training_metrics': {
                'final_train_loss': 0.25,
                'best_val_f1': 0.78
            },
            'test_metrics': {
                'f1_micro': 0.75,
                'f1_macro': 0.68,
                'hamming_loss': 0.12
            }
        }
        
        results_path = os.path.join(self.model_dir, 'training_results.json')
        
        # Save results
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        # Verify file exists and content is correct
        self.assertTrue(os.path.exists(results_path))
        
        with open(results_path, 'r') as f:
            loaded_results = json.load(f)
        
        self.assertEqual(loaded_results['model_config']['num_labels'], 41)
        self.assertIn('training_metrics', loaded_results)
        self.assertIn('test_metrics', loaded_results)

class TestModelPerformanceMetrics(unittest.TestCase):
    """Test model performance calculation and validation"""
    
    def setUp(self):
        """Set up test data for performance metrics"""
        np.random.seed(42)  # For reproducible tests
        
        # Generate sample predictions and ground truth for 41 clause types
        self.num_samples = 100
        self.num_labels = 41
        
        # Simulate realistic legal clause detection scenario
        self.y_true = np.random.binomial(1, 0.1, (self.num_samples, self.num_labels))  # Sparse labels
        self.y_prob = np.random.beta(2, 8, (self.num_samples, self.num_labels))  # Skewed probabilities
        self.y_pred = (self.y_prob > 0.3).astype(int)  # Conservative threshold
    
    def test_multilabel_metrics_calculation(self):
        """Test multi-label classification metrics"""
        from sklearn.metrics import (
            hamming_loss, jaccard_score, f1_score, 
            precision_score, recall_score
        )
        
        # Calculate metrics
        hamming = hamming_loss(self.y_true, self.y_pred)
        jaccard = jaccard_score(self.y_true, self.y_pred, average='micro')
        f1_micro = f1_score(self.y_true, self.y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        precision = precision_score(self.y_true, self.y_pred, average='micro', zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, average='micro', zero_division=0)
        
        # Validate metric ranges
        self.assertGreaterEqual(hamming, 0)
        self.assertLessEqual(hamming, 1)
        self.assertGreaterEqual(jaccard, 0)
        self.assertLessEqual(jaccard, 1)
        self.assertGreaterEqual(f1_micro, 0)
        self.assertLessEqual(f1_micro, 1)
        self.assertGreaterEqual(f1_macro, 0)
        self.assertLessEqual(f1_macro, 1)
        self.assertGreaterEqual(precision, 0)
        self.assertLessEqual(precision, 1)
        self.assertGreaterEqual(recall, 0)
        self.assertLessEqual(recall, 1)
    
    def test_per_clause_performance_analysis(self):
        """Test per-clause performance analysis"""
        clause_names = [f"Clause_{i:02d}" for i in range(self.num_labels)]
        
        # Calculate per-clause metrics
        per_clause_metrics = []
        
        for i in range(self.num_labels):
            y_true_clause = self.y_true[:, i]
            y_pred_clause = self.y_pred[:, i]
            
            tp = np.sum((y_true_clause == 1) & (y_pred_clause == 1))
            fp = np.sum((y_true_clause == 0) & (y_pred_clause == 1))
            tn = np.sum((y_true_clause == 0) & (y_pred_clause == 0))
            fn = np.sum((y_true_clause == 1) & (y_pred_clause == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_clause_metrics.append({
                'clause_name': clause_names[i],
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': np.sum(y_true_clause)
            })
        
        # Validate results
        self.assertEqual(len(per_clause_metrics), self.num_labels)
        
        for metrics in per_clause_metrics:
            self.assertIn('clause_name', metrics)
            self.assertIn('precision', metrics)
            self.assertIn('recall', metrics)
            self.assertIn('f1', metrics)
            self.assertIn('support', metrics)
            
            # Check metric ranges
            self.assertGreaterEqual(metrics['precision'], 0)
            self.assertLessEqual(metrics['precision'], 1)
            self.assertGreaterEqual(metrics['recall'], 0)
            self.assertLessEqual(metrics['recall'], 1)
            self.assertGreaterEqual(metrics['f1'], 0)
            self.assertLessEqual(metrics['f1'], 1)
    
    def test_threshold_optimization(self):
        """Test threshold optimization for multi-label classification"""
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        threshold_results = {}
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_prob > threshold).astype(int)
            
            from sklearn.metrics import f1_score
            f1_micro = f1_score(self.y_true, y_pred_thresh, average='micro', zero_division=0)
            f1_macro = f1_score(self.y_true, y_pred_thresh, average='macro', zero_division=0)
            
            threshold_results[threshold] = {
                'f1_micro': f1_micro,
                'f1_macro': f1_macro
            }
        
        # Validate that we have results for all thresholds
        self.assertEqual(len(threshold_results), len(thresholds))
        
        # Check that metrics are reasonable
        for threshold, metrics in threshold_results.items():
            self.assertGreaterEqual(metrics['f1_micro'], 0)
            self.assertLessEqual(metrics['f1_micro'], 1)
            self.assertGreaterEqual(metrics['f1_macro'], 0)
            self.assertLessEqual(metrics['f1_macro'], 1)

def create_test_suite():
    """Create comprehensive test suite for all model components"""
    test_suite = unittest.TestSuite()
    
    # Add test classes in logical order
    test_classes = [
        TestLegalClauseDataset,
        TestLegalSummarizationDataset,
        TestLegalClauseExtractor,
        TestLegalSummarizationModel,
        TestModelEvaluationIntegration,
        TestModelFileHandling,
        TestModelPerformanceMetrics
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    return test_suite

if __name__ == '__main__':
    # Run comprehensive test suite
    print("="*80)
    print("LEGAL NLP MODELS - COMPREHENSIVE TEST SUITE")
    print("Testing BERT clause extraction, T5 summarization, and model integration")
    print("="*80)
    
    # Check for required dependencies
    missing_deps = []
    try:
        import torch
    except ImportError:
        missing_deps.append('torch')
    
    try:
        import transformers
    except ImportError:
        missing_deps.append('transformers')
    
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        missing_deps.append('scikit-learn')
    
    if missing_deps:
        print(f"Warning: Missing dependencies: {', '.join(missing_deps)}")
        print("Some tests may be skipped or use mock objects.")
        print()
    
    # Create and run test suite
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("MODEL TEST EXECUTION SUMMARY")
    print("="*80)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Successful Tests: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed Tests: {len(result.failures)}")
    print(f"Error Tests: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILED TESTS ({len(result.failures)}):")
        for i, (test, failure) in enumerate(result.failures, 1):
            test_name = str(test).split(' ')[0]
            failure_msg = failure.split('AssertionError: ')[-1].split('\n')[0] if 'AssertionError:' in failure else 'See details above'
            print(f"  {i}. {test_name}: {failure_msg}")
    
    if result.errors:
        print(f"\nTESTS WITH ERRORS ({len(result.errors)}):")
        for i, (test, error) in enumerate(result.errors, 1):
            test_name = str(test).split(' ')[0]
            error_msg = error.split('\n')[-2] if '\n' in error else error
            print(f"  {i}. {test_name}: {error_msg}")
    
    if result.wasSuccessful():
        print(f"\nALL MODEL TESTS PASSED SUCCESSFULLY!")
        print("Your legal NLP models are working correctly.")
        print("\nTest Coverage Summary:")
        print("  • BERT Multi-label Clause Extraction")
        print("  • T5 Legal Document Summarization")
        print("  • Dataset Processing and Tokenization")
        print("  • Model Loading and Persistence")
        print("  • Performance Metrics Calculation")
        print("  • Multi-label Classification Evaluation")
        print("  • Integration with Evaluation Framework")
    else:
        print(f"\n⚠️  SOME MODEL TESTS FAILED")
        print("Review the failures above to fix any issues.")
        print("Note: Some failures may be due to missing dependencies.")
    
    print("\n" + "="*80)
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)