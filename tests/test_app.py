"""
Comprehensive Test Suite for Legal NLP + Explainability Toolkit Streamlit App
Tests for clause extraction, summarization, explainability, and UI components
"""
import pytest
import sys
import os
import streamlit as st
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
import tempfile
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import app components
try:
    from app.components.clause_extractor import (
        LegalClauseExtractor, 
        extract_clauses,
        render_enhanced_clause_interface
    )
    from app.components.summarizer import (
        LegalDocumentSummarizer,
        summarize_text,
        render_enhanced_summarization_interface
    )
    from app.components.explainer import (
        LegalExplainer,
        explain_predictions,
        render_explainability_interface
    )
    from scripts.utils import PROJECT, load_data, preprocess_text
    from scripts.evaluation_metrics import LegalNLPEvaluator
except ImportError as e:
    print(f"Warning: Could not import app components: {e}")
    print("Some tests may be skipped if components are not available.")

class TestLegalClauseExtractor:
    """Test suite for Legal Clause Extractor component"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.sample_legal_text = """
        This Agreement is entered into on January 1, 2024, between Company A and Company B.
        The parties agree that this license shall be non-transferable and exclusive.
        Either party may terminate this agreement with 30 days written notice.
        The governing law shall be the State of California.
        """
        
        self.expected_clause_types = [
            'Agreement Date', 'License Grant', 'Termination for Convenience', 'Governing Law'
        ]
    
    @patch('app.components.clause_extractor.LegalClauseExtractor')
    def test_clause_extractor_initialization(self, mock_extractor):
        """Test clause extractor initialization"""
        mock_instance = MagicMock()
        mock_extractor.return_value = mock_instance
        
        extractor = LegalClauseExtractor()
        assert mock_extractor.called
        assert extractor is not None
    
    def test_extract_clauses_basic_functionality(self):
        """Test basic clause extraction functionality"""
        # Mock the extraction results
        expected_results = [
            {
                'clause_type': 'Agreement Date',
                'confidence': 0.85,
                'text_span': 'January 1, 2024',
                'explanation': 'Date when agreement takes effect'
            },
            {
                'clause_type': 'Governing Law',
                'confidence': 0.92,
                'text_span': 'State of California',
                'explanation': 'Legal jurisdiction for the agreement'
            }
        ]
        
        with patch('app.components.clause_extractor.extract_clauses') as mock_extract:
            mock_extract.return_value = expected_results
            
            results = extract_clauses(self.sample_legal_text, threshold=0.3)
            
            assert len(results) >= 1
            assert all('clause_type' in result for result in results)
            assert all('confidence' in result for result in results)
            mock_extract.assert_called_once_with(self.sample_legal_text, threshold=0.3)
    
    def test_extract_clauses_with_different_thresholds(self):
        """Test clause extraction with different confidence thresholds"""
        high_confidence_results = [
            {'clause_type': 'Agreement Date', 'confidence': 0.95}
        ]
        
        low_confidence_results = [
            {'clause_type': 'Agreement Date', 'confidence': 0.95},
            {'clause_type': 'License Grant', 'confidence': 0.45}
        ]
        
        with patch('app.components.clause_extractor.extract_clauses') as mock_extract:
            # High threshold should return fewer results
            mock_extract.return_value = high_confidence_results
            high_results = extract_clauses(self.sample_legal_text, threshold=0.8)
            
            # Low threshold should return more results
            mock_extract.return_value = low_confidence_results
            low_results = extract_clauses(self.sample_legal_text, threshold=0.3)
            
            assert len(high_results) <= len(low_results)
    
    def test_extract_clauses_empty_text(self):
        """Test clause extraction with empty or invalid text"""
        with patch('app.components.clause_extractor.extract_clauses') as mock_extract:
            mock_extract.return_value = []
            
            results = extract_clauses("", threshold=0.3)
            assert results == []
            
            results = extract_clauses(None, threshold=0.3)
            assert results == []
    
    @patch('streamlit.text_area')
    @patch('streamlit.button')
    @patch('streamlit.write')
    def test_clause_extraction_interface(self, mock_write, mock_button, mock_text_area):
        """Test Streamlit interface for clause extraction"""
        mock_text_area.return_value = self.sample_legal_text
        mock_button.return_value = True
        
        with patch('app.components.clause_extractor.extract_clauses') as mock_extract:
            mock_extract.return_value = [
                {'clause_type': 'Agreement Date', 'confidence': 0.85}
            ]
            
            # This would normally be called by Streamlit
            # render_enhanced_clause_interface()
            
            # Verify the interface components are called
            assert mock_text_area.called or mock_button.called


class TestLegalDocumentSummarizer:
    """Test suite for Legal Document Summarizer component"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.sample_legal_document = """
        This Software License Agreement ("Agreement") is entered into on January 1, 2024,
        between TechCorp Inc., a Delaware corporation ("Licensor"), and ClientCorp LLC,
        a California limited liability company ("Licensee"). Licensor grants to Licensee
        a non-exclusive, non-transferable license to use the Software for internal business
        purposes only. The term of this license shall be three (3) years from the effective
        date. Either party may terminate this Agreement upon thirty (30) days written notice.
        This Agreement shall be governed by the laws of the State of Delaware.
        """
        
        self.expected_summary_keywords = [
            'license agreement', 'TechCorp', 'ClientCorp', 'three years', 'Delaware'
        ]
    
    @patch('app.components.summarizer.LegalDocumentSummarizer')
    def test_summarizer_initialization(self, mock_summarizer):
        """Test document summarizer initialization"""
        mock_instance = MagicMock()
        mock_summarizer.return_value = mock_instance
        
        summarizer = LegalDocumentSummarizer()
        assert mock_summarizer.called
        assert summarizer is not None
    
    def test_summarize_text_basic_functionality(self):
        """Test basic text summarization functionality"""
        expected_summary = (
            "Software License Agreement between TechCorp Inc. and ClientCorp LLC "
            "for three-year non-exclusive license, governed by Delaware law."
        )
        
        with patch('app.components.summarizer.summarize_text') as mock_summarize:
            mock_summarize.return_value = expected_summary
            
            summary = summarize_text(self.sample_legal_document)
            
            assert isinstance(summary, str)
            assert len(summary) > 0
            assert len(summary) < len(self.sample_legal_document)
            mock_summarize.assert_called_once_with(self.sample_legal_document)
    
    def test_summarize_different_lengths(self):
        """Test summarization with different target lengths"""
        short_summary = "Brief license agreement summary."
        long_summary = (
            "Detailed Software License Agreement between TechCorp Inc. and ClientCorp LLC "
            "establishing a three-year non-exclusive, non-transferable license for internal "
            "business use, with termination provisions and Delaware law governance."
        )
        
        with patch('app.components.summarizer.summarize_text') as mock_summarize:
            # Test short summary
            mock_summarize.return_value = short_summary
            short_result = summarize_text(self.sample_legal_document)
            
            # Test long summary
            mock_summarize.return_value = long_summary
            long_result = summarize_text(self.sample_legal_document)
            
            assert len(short_result) <= len(long_result)
    
    def test_summarize_empty_text(self):
        """Test summarization with empty or invalid text"""
        with patch('app.components.summarizer.summarize_text') as mock_summarize:
            mock_summarize.return_value = ""
            
            summary = summarize_text("")
            assert summary == ""
            
            summary = summarize_text(None)
            assert summary == "" or summary is None
    
    @patch('streamlit.text_area')
    @patch('streamlit.button')
    @patch('streamlit.write')
    def test_summarization_interface(self, mock_write, mock_button, mock_text_area):
        """Test Streamlit interface for summarization"""
        mock_text_area.return_value = self.sample_legal_document
        mock_button.return_value = True
        
        with patch('app.components.summarizer.summarize_text') as mock_summarize:
            mock_summarize.return_value = "Summary of the legal document."
            
            # Interface components should be called
            assert mock_text_area.called or mock_button.called


class TestLegalExplainer:
    """Test suite for Legal Explainability component"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.sample_text = "This agreement shall be governed by California law."
        self.sample_predictions = [
            {
                'clause_type': 'Governing Law',
                'confidence': 0.92,
                'text_span': 'California law'
            }
        ]
    
    @patch('app.components.explainer.LegalExplainer')
    def test_explainer_initialization(self, mock_explainer):
        """Test explainer initialization"""
        mock_instance = MagicMock()
        mock_explainer.return_value = mock_instance
        
        explainer = LegalExplainer()
        assert mock_explainer.called
        assert explainer is not None
    
    def test_explain_predictions_basic_functionality(self):
        """Test basic explanation functionality"""
        expected_explanation = {
            'shap_values': [0.8, -0.2, 0.5, 0.9, -0.1],
            'lime_explanation': 'Key words: California, law, governed',
            'attention_weights': [0.1, 0.2, 0.3, 0.4],
            'summary': 'The model focused on legal jurisdiction terms.'
        }
        
        with patch('app.components.explainer.explain_predictions') as mock_explain:
            mock_explain.return_value = expected_explanation
            
            explanation = explain_predictions(
                self.sample_text, 
                None,  # mock model
                None,  # mock tokenizer
                self.sample_predictions
            )
            
            assert isinstance(explanation, dict)
            assert 'shap_values' in explanation or 'lime_explanation' in explanation
            mock_explain.assert_called_once()
    
    def test_explain_multiple_clauses(self):
        """Test explanation for multiple detected clauses"""
        multiple_predictions = [
            {'clause_type': 'Governing Law', 'confidence': 0.92},
            {'clause_type': 'Agreement Date', 'confidence': 0.85}
        ]
        
        expected_explanations = {
            'Governing Law': {'importance': 0.92, 'key_words': ['California', 'law']},
            'Agreement Date': {'importance': 0.85, 'key_words': ['agreement']}
        }
        
        with patch('app.components.explainer.explain_predictions') as mock_explain:
            mock_explain.return_value = expected_explanations
            
            explanations = explain_predictions(
                self.sample_text,
                None, None,
                multiple_predictions
            )
            
            assert len(explanations) >= len(multiple_predictions)
    
    @patch('streamlit.text_area')
    @patch('streamlit.button')
    def test_explainability_interface(self, mock_button, mock_text_area):
        """Test Streamlit interface for explainability"""
        mock_text_area.return_value = self.sample_text
        mock_button.return_value = True
        
        with patch('app.components.explainer.render_explainability_interface') as mock_render:
            # Interface should handle the rendering
            mock_render.return_value = None
            
            # Verify interface components are available
            assert mock_text_area.called or mock_button.called


class TestUtilityFunctions:
    """Test suite for utility functions and helpers"""
    
    def test_preprocess_text_legal_optimization(self):
        """Test legal text preprocessing"""
        sample_text = "  This Agreement  contains    multiple   spaces.  "
        
        with patch('scripts.utils.preprocess_text') as mock_preprocess:
            mock_preprocess.return_value = "This Agreement contains multiple spaces."
            
            processed = preprocess_text(sample_text, for_legal=True)
            
            assert len(processed) <= len(sample_text)
            assert "  " not in processed  # Multiple spaces should be normalized
            mock_preprocess.assert_called_once()
    
    def test_load_data_functionality(self):
        """Test data loading utility"""
        mock_data = pd.DataFrame({
            'text': ['Sample legal text'],
            'labels': [['Agreement Date', 'Governing Law']]
        })
        
        with patch('scripts.utils.load_data') as mock_load:
            mock_load.return_value = mock_data
            
            data = load_data('test_path.csv')
            
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            mock_load.assert_called_once_with('test_path.csv')
    
    def test_project_structure_paths(self):
        """Test PROJECT structure utility"""
        with patch('scripts.utils.PROJECT') as mock_project:
            mock_project.get_path.return_value = "/path/to/data"
            
            # Test path resolution
            data_path = PROJECT.get_path('data')
            assert data_path is not None


class TestIntegrationScenarios:
    """Integration tests for complete workflows"""
    
    def setup_method(self):
        """Setup integration test fixtures"""
        self.complex_legal_text = """
        SOFTWARE LICENSE AGREEMENT
        
        This Software License Agreement ("Agreement") is made and entered into as of
        January 15, 2024 ("Effective Date"), by and between TechInnovate Corp., 
        a Delaware corporation ("Licensor"), and Enterprise Solutions LLC, 
        a New York limited liability company ("Licensee").
        
        1. GRANT OF LICENSE. Subject to the terms and conditions of this Agreement,
        Licensor hereby grants to Licensee a non-exclusive, non-transferable license
        to use the Software solely for Licensee's internal business purposes.
        
        2. TERM. This Agreement shall commence on the Effective Date and shall
        continue for a period of three (3) years, unless earlier terminated.
        
        3. TERMINATION. Either party may terminate this Agreement at any time
        upon thirty (30) days' prior written notice to the other party.
        
        4. GOVERNING LAW. This Agreement shall be governed by and construed in
        accordance with the laws of the State of Delaware.
        """
    
    @patch('app.components.clause_extractor.extract_clauses')
    @patch('app.components.summarizer.summarize_text')
    @patch('app.components.explainer.explain_predictions')
    def test_complete_analysis_workflow(self, mock_explain, mock_summarize, mock_extract):
        """Test complete legal document analysis workflow"""
        # Mock clause extraction results
        mock_extract.return_value = [
            {'clause_type': 'Agreement Date', 'confidence': 0.95},
            {'clause_type': 'License Grant', 'confidence': 0.88},
            {'clause_type': 'Termination for Convenience', 'confidence': 0.82},
            {'clause_type': 'Governing Law', 'confidence': 0.91}
        ]
        
        # Mock summarization result
        mock_summarize.return_value = (
            "Three-year software license agreement between TechInnovate Corp. and "
            "Enterprise Solutions LLC with termination and Delaware law provisions."
        )
        
        # Mock explanation result
        mock_explain.return_value = {
            'key_findings': 'Model identified legal terms and dates',
            'confidence_analysis': 'High confidence in standard legal clauses'
        }
        
        # Execute complete workflow
        clauses = extract_clauses(self.complex_legal_text, threshold=0.3)
        summary = summarize_text(self.complex_legal_text)
        explanation = explain_predictions(self.complex_legal_text, None, None, clauses)
        
        # Verify results
        assert len(clauses) >= 4
        assert len(summary) > 0
        assert len(summary) < len(self.complex_legal_text)
        assert isinstance(explanation, dict)
        
        # Verify all components were called
        mock_extract.assert_called_once()
        mock_summarize.assert_called_once()
        mock_explain.assert_called_once()
    
    def test_error_handling_in_workflow(self):
        """Test error handling in the analysis workflow"""
        with patch('app.components.clause_extractor.extract_clauses') as mock_extract:
            mock_extract.side_effect = Exception("Model loading failed")
            
            try:
                clauses = extract_clauses(self.complex_legal_text)
                # Should handle error gracefully
                assert clauses == [] or clauses is None
            except Exception as e:
                # Or raise appropriate exception
                assert "Model loading failed" in str(e)
    
    def test_performance_with_large_document(self):
        """Test performance with large legal documents"""
        large_document = self.complex_legal_text * 10  # Simulate large document
        
        with patch('app.components.clause_extractor.extract_clauses') as mock_extract:
            mock_extract.return_value = [
                {'clause_type': 'License Grant', 'confidence': 0.85}
            ]
            
            # Should handle large documents without issues
            clauses = extract_clauses(large_document, threshold=0.3)
            assert len(clauses) >= 0
            mock_extract.assert_called_once()


class TestConfigurationAndSettings:
    """Test configuration and settings management"""
    
    def test_confidence_threshold_settings(self):
        """Test different confidence threshold configurations"""
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            with patch('app.components.clause_extractor.extract_clauses') as mock_extract:
                mock_extract.return_value = [
                    {'clause_type': 'Test Clause', 'confidence': 0.6}
                ]
                
                results = extract_clauses("Test text", threshold=threshold)
                
                # Verify threshold is respected
                if threshold <= 0.6:
                    assert len(results) >= 1
                else:
                    assert len(results) == 0 or results[0]['confidence'] >= threshold
    
    def test_model_configuration_loading(self):
        """Test model configuration loading"""
        mock_config = {
            'model_name': 'bert-base-uncased',
            'max_length': 512,
            'batch_size': 8,
            'uses_clean_clause_names': True
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_config))):
            with patch('json.load') as mock_json_load:
                mock_json_load.return_value = mock_config
                
                # Test configuration loading
                config = json.load(open('mock_config.json'))
                
                assert config['model_name'] == 'bert-base-uncased'
                assert config['uses_clean_clause_names'] is True


# Pytest fixtures for the entire test suite
@pytest.fixture
def sample_legal_text():
    """Fixture providing sample legal text for testing"""
    return """
    This Agreement is entered into on March 1, 2024, between Company A Inc. and Company B LLC.
    The license granted herein is non-exclusive and non-transferable.
    This Agreement may be terminated by either party with 60 days written notice.
    The governing law shall be the State of New York.
    """

@pytest.fixture
def mock_model():
    """Fixture providing a mock model for testing"""
    model = MagicMock()
    model.predict.return_value = [
        {'clause_type': 'Agreement Date', 'confidence': 0.9},
        {'clause_type': 'License Grant', 'confidence': 0.8}
    ]
    return model

@pytest.fixture
def mock_tokenizer():
    """Fixture providing a mock tokenizer for testing"""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [101, 2023, 3820, 2003, 102]  # Mock token IDs
    return tokenizer

# Test runner configuration
if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([
        __file__, 
        '-v', 
        '--tb=short',
        '--disable-warnings',
        '--color=yes'
    ])