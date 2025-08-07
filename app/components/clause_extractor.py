"""
Legal Clause Extractor Component for Streamlit App
Enhanced multi-label BERT clause detection with 41 CUAD clause types and explainability
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from functools import lru_cache
import time

import streamlit as st
from transformers import AutoTokenizer, AutoModel
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.utils import load_data, clean_clause_name, PROJECT
from scripts.evaluation_metrics import LegalNLPEvaluator

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ClauseResult:
    """Data class for clause detection results"""
    clause_type: str
    clean_name: str
    confidence: float
    position_start: Optional[int] = None
    position_end: Optional[int] = None
    matched_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ExtractionConfig:
    """Configuration for clause extraction"""
    confidence_threshold: float = 0.3
    max_length: int = 512
    batch_size: int = 1
    return_positions: bool = False
    return_matched_text: bool = False
    enable_preprocessing: bool = True
    
class MultiLabelBERT(torch.nn.Module):
    """Enhanced multi-label BERT model for legal clause classification"""
    
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.3):
        super(MultiLabelBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels
        
        # Add batch normalization for better stability
        self.batch_norm = torch.nn.BatchNorm1d(self.bert.config.hidden_size)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use pooler output with batch normalization
        pooled_output = outputs.pooler_output
        pooled_output = self.batch_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        result = {
            'logits': logits,
            'hidden_states': outputs.last_hidden_state,
            'attention_weights': outputs.attentions if hasattr(outputs, 'attentions') else None
        }
        
        if labels is not None:
            # Calculate multi-label loss
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
            result['loss'] = loss
            
        return result

class LegalClauseExtractor:
    """
    Enhanced Legal Clause Extractor with improved performance and features
    """
    
    # Complete CUAD clause types from actual training data (41 types)
    DEFAULT_CLAUSE_TYPES = [
        "Third Party Beneficiary", "Renewal Term", "Cap On Liability", "Warranty Duration",
        "Audit Rights", "No-Solicit Of Employees", "Exclusivity", "Effective Date",
        "No-Solicit Of Customers", "License Grant", "Uncapped Liability", "Non-Disparagement",
        "Price Restrictions", "Competitive Restriction Exception", "Source Code Escrow",
        "Irrevocable Or Perpetual License", "Covenant Not To Sue", "Minimum Commitment",
        "Liquidated Damages", "Change Of Control", "Revenue/Profit Sharing",
        "Affiliate License-Licensor", "Ip Ownership Assignment", "Parties",
        "Governing Law", "Agreement Date", "Joint Ip Ownership", "Post-Termination Services",
        "Termination For Convenience", "Expiration Date", "Notice Period To Terminate Renewal",
        "Rofr/Rofo/Rofn", "Volume Restriction", "Non-Compete", "Affiliate License-Licensee",
        "Non-Transferable License", "Insurance", "Unlimited/All-You-Can-Eat-License",
        "Most Favored Nation", "Document Name", "Anti-Assignment"
    ]
    
    def __init__(self, model_path: str = None, cache_dir: str = None, config: ExtractionConfig = None):
        """
        Initialize the Enhanced Legal Clause Extractor
        
        Args:
            model_path: Path to trained BERT model directory
            cache_dir: Directory for caching models and data
            config: Configuration for extraction parameters
        """
        self.config = config or ExtractionConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.clause_types = []
        self.clean_clause_names = {}
        self.evaluator = None
        self._model_loaded = False
        self._performance_metrics = {}
        
        # Enhanced path management
        self.project_root = Path(__file__).parent.parent.parent
        self.model_path = Path(model_path) if model_path else self.project_root / 'models' / 'bert'
        self.cache_dir = Path(cache_dir) if cache_dir else self.project_root / 'app' / '.cache'
        
        # Log initialization info
        logger.info(f"🔧 Initializing Legal Clause Extractor")
        logger.info(f"   Project root: {self.project_root}")
        logger.info(f"   Model path: {self.model_path}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Model path exists: {self.model_path.exists()}")
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and metadata
        self._initialize()
    
    def _initialize(self):
        """Initialize model, metadata, and evaluator"""
        try:
            start_time = time.time()
            
            # Load metadata first
            self._load_metadata()
            
            # Load model
            self._load_model()
            
            # Initialize evaluator
            self._initialize_evaluator()
            
            init_time = time.time() - start_time
            self._performance_metrics['initialization_time'] = init_time
            logger.info(f"Clause extractor initialized in {init_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            self._load_fallback()
    
    def _load_metadata(self):
        """Load clause metadata with fallback to defaults"""
        # First try to load from clean_clause_names.json (which has the correct mapping)
        clean_names_path = self.project_root / 'models' / 'bert' / 'clean_clause_names.json'
        metadata_path = self.project_root / 'data' / 'processed' / 'metadata.json'
        
        try:
            if clean_names_path.exists():
                logger.info(f"📄 Loading clause names from {clean_names_path}")
                with open(clean_names_path, 'r', encoding='utf-8') as f:
                    clean_data = json.load(f)
                
                if 'clean_clause_names' in clean_data:
                    self.clause_types = clean_data['clean_clause_names']
                    logger.info(f"✅ Loaded {len(self.clause_types)} clause types from clean_clause_names.json")
                
                # Create clean mapping
                if 'original_to_clean_mapping' in clean_data:
                    self.clean_clause_names = clean_data['original_to_clean_mapping']
                    # Also create direct mapping for clean names
                    for clean_name in self.clause_types:
                        self.clean_clause_names[clean_name] = clean_name
                else:
                    # Create clean mapping from clause types
                    self.clean_clause_names = {clause_type: clause_type for clause_type in self.clause_types}
                
                logger.info(f"✅ Created mapping for {len(self.clean_clause_names)} clause names")
                return
                
            elif metadata_path.exists():
                logger.info(f"📄 Loading metadata from {metadata_path}")
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.clause_types = metadata.get('clause_types', self.DEFAULT_CLAUSE_TYPES)
                self.clean_clause_names = metadata.get('clean_clause_names', {})
                
                # Ensure clean names exist for all clause types
                for clause_type in self.clause_types:
                    if clause_type not in self.clean_clause_names:
                        self.clean_clause_names[clause_type] = clean_clause_name(clause_type)
                
                logger.info(f"✅ Loaded metadata for {len(self.clause_types)} clause types")
            else:
                logger.warning(f"❌ No metadata files found at {clean_names_path} or {metadata_path}")
                self._create_default_metadata()
                
        except Exception as e:
            logger.error(f"❌ Error loading metadata: {e}")
            self._create_default_metadata()
    
    def _create_default_metadata(self):
        """Create comprehensive default metadata"""
        self.clause_types = self.DEFAULT_CLAUSE_TYPES
        self.clean_clause_names = {
            clause_type: clean_clause_name(clause_type) 
            for clause_type in self.clause_types
        }
        logger.info(f"Created default metadata for {len(self.clause_types)} clause types")
    
    def _load_model(self):
        """Load model with enhanced error handling and fallback"""
        try:
            if self.model_path.exists():
                self._load_trained_model()
            else:
                logger.warning(f"Model directory not found at {self.model_path}")
                self._load_default_model()
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._load_default_model()
    
    def _load_trained_model(self):
        """Load fine-tuned model with comprehensive configuration"""
        try:
            # Load training configuration
            config_path = self.model_path / 'config.json'
            training_results_path = self.model_path / 'training_results.json'
            
            model_name = 'nlpaueb/legal-bert-base-uncased'  # Use the actual trained model
            num_labels = len(self.clause_types)
            
            if training_results_path.exists():
                with open(training_results_path, 'r') as f:
                    training_results = json.load(f)
                    
                model_config = training_results.get('model_config', {})
                model_name = model_config.get('model_name', model_name)
                num_labels = model_config.get('num_labels', num_labels)
                
                # Ensure we have the correct number of clause types from training
                if 'clean_clause_names' in training_results:
                    trained_clause_names = training_results['clean_clause_names']
                    if len(trained_clause_names) == 41:
                        self.clause_types = trained_clause_names
                        logger.info(f"Loaded {len(trained_clause_names)} clause types from training results")
            
            # Initialize model
            self.model = MultiLabelBERT(model_name, num_labels)
            
            # Load trained weights
            model_weights_path = self.model_path / 'final_model.pt'
            if model_weights_path.exists():
                state_dict = torch.load(model_weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info("Loaded fine-tuned model weights")
            
            # Load tokenizer
            tokenizer_path = self.model_path / 'tokenizer'
            if tokenizer_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
            
            logger.info(f"Successfully loaded trained model with {num_labels} labels")
            
        except Exception as e:
            logger.error(f"Error loading trained model: {e}")
            self._load_default_model()
    
    def _load_default_model(self):
        """Load default BERT model with error handling"""
        try:
            model_name = 'nlpaueb/legal-bert-base-uncased'  # Use the same model as training
            num_labels = len(self.clause_types)
            
            self.model = MultiLabelBERT(model_name, num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
            
            logger.info(f"Loaded default Legal-BERT model with {num_labels} labels")
            
        except Exception as e:
            logger.error(f"Critical error loading default model: {e}")
            self._load_fallback()
    
    def _load_fallback(self):
        """Load minimal fallback for basic functionality"""
        logger.warning("Loading fallback mode - pattern-based detection only")
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        
        if not self.clause_types:
            self._create_default_metadata()
    
    def _initialize_evaluator(self):
        """Initialize evaluation component"""
        try:
            self.evaluator = LegalNLPEvaluator(
                clause_types=self.clause_types,
                clean_clause_names=self.clean_clause_names
            )
        except Exception as e:
            logger.warning(f"Could not initialize evaluator: {e}")
            self.evaluator = None
    
    @lru_cache(maxsize=128)
    def _preprocess_text_cached(self, text: str) -> str:
        """Cached text preprocessing for better performance"""
        return self._preprocess_text_impl(text)
    
    def _preprocess_text_impl(self, text: str) -> str:
        """Enhanced legal text preprocessing"""
        if not self.config.enable_preprocessing:
            return text.strip()
        
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Legal-specific preprocessing
        legal_replacements = {
            r'\bhereof\b': 'of this agreement',
            r'\bthereof\b': 'of that',
            r'\bherein\b': 'in this agreement',
            r'\bwherein\b': 'in which',
            r'\bwhereby\b': 'by which',
            r'\bheretofore\b': 'before this',
            r'\bhereinafter\b': 'after this',
            r'\bnotwithstanding\b': 'despite',
            r'\bpursuant to\b': 'according to',
        }
        
        for pattern, replacement in legal_replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Clean up excessive punctuation
        text = re.sub(r'[.,;:]{2,}', '.', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def extract_clauses(self, 
                       text: Union[str, List[str]], 
                       config: Optional[ExtractionConfig] = None) -> Dict[str, Any]:
        """
        Enhanced clause extraction with batch processing support
        
        Args:
            text: Input text(s) - single string or list of strings
            config: Configuration override for this extraction
            
        Returns:
            Comprehensive results dictionary
        """
        # Use provided config or instance config
        extraction_config = config or self.config
        
        # Handle empty input
        if not text or (isinstance(text, str) and not text.strip()):
            return self._create_empty_result("Empty input text", extraction_config)
        
        # Convert single text to list for uniform processing
        texts = [text] if isinstance(text, str) else text
        
        start_time = time.time()
        
        try:
            if not self._model_loaded:
                return self._fallback_extraction(texts[0], extraction_config)
            
            # Process texts
            results = self._extract_from_texts(texts, extraction_config)
            
            # Add performance metrics
            processing_time = time.time() - start_time
            results['processing_info']['processing_time'] = processing_time
            results['processing_info']['texts_processed'] = len(texts)
            results['processing_info']['avg_time_per_text'] = processing_time / len(texts)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during clause extraction: {e}")
            return self._create_error_result(str(e), extraction_config, len(texts))
    
    def _extract_from_texts(self, texts: List[str], config: ExtractionConfig) -> Dict[str, Any]:
        """Process multiple texts with batching support"""
        all_results = []
        total_clauses = 0
        all_confidences = []
        
        for text in texts:
            # Preprocess
            processed_text = self._preprocess_text_cached(text) if config.enable_preprocessing else text
            
            # Tokenize
            encoding = self.tokenizer(
                processed_text,
                truncation=True,
                padding='max_length',
                max_length=config.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
            
            # Process predictions for this text
            text_results = self._process_predictions(
                probabilities, 
                config, 
                original_text=text if config.return_matched_text else None
            )
            
            all_results.extend(text_results)
            total_clauses += len(text_results)
            all_confidences.extend([r.confidence for r in text_results])
        
        # Aggregate results
        return self._aggregate_results(all_results, all_confidences, config, texts)
    
    def _process_predictions(self, 
                           probabilities: np.ndarray, 
                           config: ExtractionConfig,
                           original_text: Optional[str] = None) -> List[ClauseResult]:
        """Enhanced prediction processing with position tracking"""
        results = []
        
        for i, prob in enumerate(probabilities):
            if prob > config.confidence_threshold:
                clause_type = self.clause_types[i]
                clean_name = self.clean_clause_names.get(clause_type, clause_type)
                
                clause_result = ClauseResult(
                    clause_type=clause_type,
                    clean_name=clean_name,
                    confidence=float(prob)
                )
                
                # Add position and matched text if requested
                if config.return_positions and original_text:
                    positions = self._find_clause_positions(original_text, clause_type)
                    if positions:
                        clause_result.position_start = positions[0]
                        clause_result.position_end = positions[1]
                
                if config.return_matched_text and original_text and hasattr(clause_result, 'position_start'):
                    if clause_result.position_start is not None:
                        clause_result.matched_text = original_text[
                            clause_result.position_start:clause_result.position_end
                        ]
                
                results.append(clause_result)
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results
    
    def _find_clause_positions(self, text: str, clause_type: str) -> Optional[Tuple[int, int]]:
        """Find approximate positions of clause text (simplified implementation)"""
        # This is a simplified implementation - in practice, you'd use more sophisticated
        # methods like attention weights or token-level predictions
        try:
            clean_name = self.clean_clause_names.get(clause_type, clause_type)
            # Look for keywords related to the clause type
            keywords = clean_name.lower().split()
            
            for keyword in keywords:
                match = re.search(rf'\b{re.escape(keyword)}\b', text, re.IGNORECASE)
                if match:
                    # Return a context window around the match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    return (start, end)
                    
        except Exception:
            pass
        
        return None
    
    def _aggregate_results(self, 
                          all_results: List[ClauseResult], 
                          all_confidences: List[float],
                          config: ExtractionConfig,
                          original_texts: List[str]) -> Dict[str, Any]:
        """Aggregate results from multiple texts"""
        # Convert results to dictionaries
        clause_dicts = [result.to_dict() for result in all_results]
        
        # Calculate statistics
        stats = self._calculate_statistics(all_confidences) if all_confidences else {}
        
        # Calculate complexity for first text (or combined if multiple)
        complexity = self._analyze_document_complexity(' '.join(original_texts))
        
        return {
            'detected_clauses': clause_dicts,
            'clause_count': len(all_results),
            'confidence_scores': {r.clean_name: r.confidence for r in all_results},
            'statistics': stats,
            'complexity_analysis': complexity,
            'processing_info': {
                'model_loaded': self._model_loaded,
                'device_used': str(self.device),
                'config_used': asdict(config),
                'total_input_length': sum(len(text) for text in original_texts)
            }
        }
    
    def _calculate_statistics(self, confidences: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        if not confidences:
            return {'total_clauses': 0, 'avg_confidence': 0.0}
        
        confidences_array = np.array(confidences)
        
        return {
            'total_clauses': len(confidences),
            'avg_confidence': float(np.mean(confidences_array)),
            'max_confidence': float(np.max(confidences_array)),
            'min_confidence': float(np.min(confidences_array)),
            'std_confidence': float(np.std(confidences_array)),
            'confidence_quartiles': {
                'q25': float(np.percentile(confidences_array, 25)),
                'q50': float(np.percentile(confidences_array, 50)),
                'q75': float(np.percentile(confidences_array, 75))
            },
            'high_confidence_count': sum(1 for c in confidences if c > 0.7),
            'medium_confidence_count': sum(1 for c in confidences if 0.3 < c <= 0.7),
            'low_confidence_count': sum(1 for c in confidences if c <= 0.3)
        }
    
    def _analyze_document_complexity(self, text: str) -> Dict[str, Any]:
        """Enhanced document complexity analysis"""
        # Basic metrics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\n\n')
        
        word_count = len(words)
        sentence_count = len([s for s in sentences if s.strip()])
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        # Advanced metrics
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Legal complexity indicators
        legal_terms = [
            'whereas', 'therefore', 'notwithstanding', 'heretofore', 'hereinafter',
            'pursuant', 'covenant', 'indemnify', 'breach', 'default', 'remedy',
            'force majeure', 'arbitration', 'jurisdiction', 'severability'
        ]
        
        legal_term_matches = sum(1 for term in legal_terms 
                                if re.search(rf'\b{term}\b', text, re.IGNORECASE))
        
        # Structural indicators
        structure_patterns = [
            r'\b(?:section|article|paragraph|subsection)\s+\d+',
            r'\b\d+\.\d+',
            r'\([a-z]\)',
            r'\b(?:exhibit|schedule|appendix)\s+[a-z]'
        ]
        
        structure_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                             for pattern in structure_patterns)
        
        # Complexity scoring
        complexity_score = (
            (legal_term_matches / max(word_count, 1)) * 100 * 2 +  # Legal terms weight
            (structure_count / max(sentence_count, 1)) * 10 +      # Structure weight
            (avg_sentence_length / 20)                             # Sentence length weight
        )
        
        complexity_level = (
            'Very High' if complexity_score > 15 else
            'High' if complexity_score > 10 else
            'Medium' if complexity_score > 5 else
            'Low'
        )
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'legal_term_count': legal_term_matches,
            'legal_term_density': round((legal_term_matches / max(word_count, 1)) * 100, 2),
            'structure_indicators': structure_count,
            'complexity_score': round(complexity_score, 2),
            'complexity_level': complexity_level,
            'readability_score': max(0, 100 - complexity_score * 2)  # Inverse relationship
        }
    
    def _fallback_extraction(self, text: str, config: ExtractionConfig) -> Dict[str, Any]:
        """Enhanced fallback extraction with better pattern matching"""
        logger.warning("Using enhanced fallback clause extraction")
        
        # Comprehensive pattern mapping for CUAD clauses
        patterns = {
            'Agreement Date': [
                r'(?:agreement|contract).*?(?:date|dated).*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(?:effective|execution).*?date.*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'this.*?agreement.*?(?:made|entered).*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            ],
            'Governing Law': [
                r'govern(?:ed|ing)\s+by.*?laws?\s+of\s+([^,.;]+)',
                r'subject\s+to.*?laws?\s+of\s+([^,.;]+)',
                r'construed\s+(?:in\s+)?accordance\s+with.*?laws?\s+of\s+([^,.;]+)'
            ],
            'Termination for Convenience': [
                r'terminat(?:e|ion|ing).*?(?:for\s+)?convenience',
                r'either\s+party.*?terminat(?:e|ion).*?without\s+cause',
                r'terminat(?:e|ion).*?at\s+will'
            ],
            'Effective Date': [
                r'effective\s+(?:as\s+of\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'commenc(?:e|ing).*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            ],
            'Expiration Date': [
                r'expir(?:e|es|ation).*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'terminat(?:e|es|ion).*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            ]
        }
        
        detected_clauses = []
        
        for clause_name, clause_patterns in patterns.items():
            max_confidence = 0.0
            best_match = None
            
            for pattern in clause_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    # Calculate confidence based on pattern strength and context
                    confidence = 0.4 + (len(match.group(0)) / 100) * 0.2
                    confidence = min(confidence, 0.8)  # Cap at 0.8 for pattern matching
                    
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_match = match.group(0)
            
            if max_confidence > config.confidence_threshold:
                clean_name = self.clean_clause_names.get(clause_name, clause_name)
                detected_clauses.append(ClauseResult(
                    clause_type=clause_name,
                    clean_name=clean_name,
                    confidence=max_confidence,
                    matched_text=best_match if config.return_matched_text else None
                ))
        
        # Sort by confidence
        detected_clauses.sort(key=lambda x: x.confidence, reverse=True)
        
        # Calculate statistics and complexity
        confidences = [c.confidence for c in detected_clauses]
        stats = self._calculate_statistics(confidences)
        complexity = self._analyze_document_complexity(text)
        
        return {
            'detected_clauses': [c.to_dict() for c in detected_clauses],
            'clause_count': len(detected_clauses),
            'confidence_scores': {c.clean_name: c.confidence for c in detected_clauses},
            'statistics': stats,
            'complexity_analysis': complexity,
            'processing_info': {
                'fallback_mode': True,
                'pattern_based': True,
                'model_loaded': False,
                'total_input_length': len(text)
            }
        }
    
    def _create_empty_result(self, error_msg: str, config: ExtractionConfig) -> Dict[str, Any]:
        """Create standardized empty result"""
        return {
            'detected_clauses': [],
            'clause_count': 0,
            'confidence_scores': {},
            'statistics': {'total_clauses': 0, 'avg_confidence': 0.0},
            'complexity_analysis': {},
            'processing_info': {
                'error': error_msg,
                'config_used': asdict(config),
                'total_input_length': 0
            }
        }
    
    def _create_error_result(self, error_msg: str, config: ExtractionConfig, text_count: int) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'detected_clauses': [],
            'clause_count': 0,
            'confidence_scores': {},
            'statistics': {'total_clauses': 0, 'avg_confidence': 0.0},
            'complexity_analysis': {},
            'processing_info': {
                'error': error_msg,
                'config_used': asdict(config),
                'texts_processed': text_count,
                'model_loaded': self._model_loaded
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'model_loaded': self._model_loaded,
            'device': str(self.device),
            'model_path': str(self.model_path),
            'num_clause_types': len(self.clause_types),
            'clause_types': self.clause_types,
            'performance_metrics': self._performance_metrics,
            'cache_dir': str(self.cache_dir),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }

# Enhanced Streamlit Interface
@st.cache_resource
def load_clause_extractor(model_path: str = None) -> Optional[LegalClauseExtractor]:
    """Load and cache the enhanced clause extractor"""
    try:
        with st.spinner("Loading legal clause extraction model..."):
            extractor = LegalClauseExtractor(model_path=model_path)
            return extractor
    except Exception as e:
        st.error(f"Failed to load clause extractor: {e}")
        logger.error(f"Extractor loading error: {e}")
        return None

def render_enhanced_clause_interface():
    """Render the enhanced Streamlit interface"""
    st.header("Enhanced Legal Clause Extraction")
    st.markdown("""
    **AI-Powered Multi-Label Classification** for 41 CUAD clause types with advanced analytics.
    Upload documents or paste text to identify legal clauses with confidence scoring.
    """)
    
    # Load extractor
    extractor = load_clause_extractor()
    if extractor is None:
        st.error("⚠️ Failed to load the clause extraction model. Please check the configuration.")
        return
    
    # Configuration sidebar (UPDATED TO MATCH SUMMARIZATION STYLE)
    st.sidebar.header("Extraction Settings")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1, max_value=0.9, value=0.3, step=0.05,
        help="Minimum confidence score for clause detection"
    )
    
    max_length = st.sidebar.selectbox(
        "Max Token Length",
        options=[256, 512, 1024],
        index=1,
        help="Maximum tokens to process (longer = more context, slower processing)"
    )
    
    advanced_features = st.sidebar.checkbox(
        "Advanced Features", 
        value=False,
        help="Enable position tracking and matched text extraction"
    )
    
    # Model Information in sidebar (CONSISTENT WITH SUMMARIZATION STYLE)
    st.sidebar.header("Model Information")
    model_info = extractor.get_model_info()
    
    # Model Status
    st.sidebar.write("**Model Status**")
    if model_info['model_loaded']:
        st.sidebar.success("✅ Loaded")
    else:
        st.sidebar.error("❌ Failed to Load")
    
    # Device
    st.sidebar.write("**Device**")
    st.sidebar.write(model_info['device'])
    
    # Model Name
    st.sidebar.write("**Clause Types**")
    st.sidebar.write(f"{model_info['num_clause_types']} types")
    
    # Initialization Time
    if 'performance_metrics' in model_info and 'initialization_time' in model_info['performance_metrics']:
        st.sidebar.write("**Init Time**")
        st.sidebar.write(f"{model_info['performance_metrics']['initialization_time']:.2f}s")
    
    # Technical Details (collapsible)
    with st.sidebar.expander("Technical Details"):
        st.json({
            'model_loaded': model_info.get('model_loaded', False),
            'num_clause_types': model_info.get('num_clause_types', 0),
            'device': model_info.get('device', 'Unknown'),
            'performance_metrics': model_info.get('performance_metrics', {})
        })
    
    # Create extraction configuration
    config = ExtractionConfig(
        confidence_threshold=confidence_threshold,
        max_length=max_length,
        return_positions=advanced_features,
        return_matched_text=advanced_features,
        enable_preprocessing=True
    )
    
    # Input section (REMOVED OLD CONFIGURATION FROM MAIN AREA)
    st.subheader(" Document Input")
    
    input_method = st.radio(
        "Choose input method:",
        options=["📝 Paste Text", "📎 Upload File", "📚 Sample Documents"],
        horizontal=True
    )
    
    text_input = ""
    
    if input_method == "📝 Paste Text":
        text_input = st.text_area(
            "Enter legal document text:",
            height=300,
            placeholder="Paste your legal document text here...",
            help="Enter the legal text you want to analyze for clause detection"
        )
    
    elif input_method == "📎 Upload File":
        uploaded_file = st.file_uploader(
            "Upload legal document",
            type=['txt'],
            help="Currently supports TXT files. PDF and DOCX support coming soon."
        )
        
        if uploaded_file is not None:
            try:
                text_input = str(uploaded_file.read(), "utf-8")
                st.success(f"✅ File loaded: {len(text_input)} characters")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    else:  # Sample Documents
        st.info("📚 Choose from sample legal documents:")
        
        sample_docs = {
            "Software License Agreement": """
            This Software License Agreement ("Agreement") is effective as of January 1, 2024 
            between TechCorp Inc., a Delaware corporation ("Licensor"), and ClientCorp LLC 
            ("Licensee"). The term of this Agreement shall be three (3) years unless terminated 
            earlier in accordance with the provisions hereof. This Agreement shall be governed 
            by the laws of the State of California. Either party may terminate this Agreement 
            for convenience upon thirty (30) days written notice. The Licensee shall pay 
            liquidated damages of $50,000 for any breach of this Agreement.
            """,
            "Service Agreement": """
            This Service Agreement is entered into on March 15, 2024, between ServiceCorp 
            and Client Inc. The agreement will expire on March 15, 2027. The governing law 
            shall be the laws of New York. ServiceCorp provides audit rights to Client Inc. 
            to examine books and records. There is a cap on liability of $100,000. 
            This agreement includes non-compete restrictions for 2 years post-termination.
            """,
            "Partnership Agreement": """
            Partnership Agreement effective April 1, 2024. The parties agree to revenue 
            sharing at 60/40 split. Joint IP ownership applies to all developments. 
            Insurance requirements include $1M liability coverage. Most favored nation 
            clauses apply to pricing. Agreement includes exclusivity provisions and 
            minimum commitment of $500K annually.
            """
        }
        
        selected_sample = st.selectbox("Select sample document:", list(sample_docs.keys()))
        if st.button("📋 Load Sample"):
            text_input = sample_docs[selected_sample]
    
    # Analysis section
    if st.button("🚀 Analyze Document", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Processing document...")
            progress_bar.progress(25)
            
            # Perform extraction
            with st.spinner("Analyzing legal clauses..."):
                results = extractor.extract_clauses(text_input, config)
            
            progress_bar.progress(75)
            status_text.text("📊 Generating results...")
            
            # Display results
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            time.sleep(0.5)  # Brief pause for UX
            progress_bar.empty()
            status_text.empty()
            
            # Results display
            st.success(f"🎯 Analysis Complete! Found **{results['clause_count']}** clauses")
            
            # Enhanced metrics display
            st.subheader("📊 Analysis Dashboard")
            
            if results['clause_count'] > 0:
                stats = results.get('statistics', {})
                complexity = results.get('complexity_analysis', {})
                
                # Main metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Clauses", stats.get('total_clauses', 0))
                with col2:
                    st.metric("Avg Confidence", f"{stats.get('avg_confidence', 0):.3f}")
                with col3:
                    st.metric("Max Confidence", f"{stats.get('max_confidence', 0):.3f}")
                with col4:
                    st.metric("Document Words", complexity.get('word_count', 0))
                with col5:
                    st.metric("Complexity", complexity.get('complexity_level', 'Unknown'))
                
                # Additional metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("High Confidence", stats.get('high_confidence_count', 0), 
                             delta="(>0.7)", delta_color="normal")
                with col2:
                    st.metric("Medium Confidence", stats.get('medium_confidence_count', 0),
                             delta="(0.3-0.7)", delta_color="normal")
                with col3:
                    st.metric("Low Confidence", stats.get('low_confidence_count', 0),
                             delta="(≤0.3)", delta_color="normal")
                
                # Detected clauses table
                st.subheader("📋 Detected Clauses")
                
                # Enhanced clause display
                clause_data = []
                for i, clause in enumerate(results['detected_clauses'], 1):
                    confidence_pct = clause['confidence'] * 100
                    
                    # Confidence color coding
                    if confidence_pct >= 70:
                        confidence_emoji = "🟢"
                    elif confidence_pct >= 30:
                        confidence_emoji = "🟡"
                    else:
                        confidence_emoji = "🔴"
                    
                    clause_info = {
                        "#": i,
                        "Clause Type": clause['clean_name'],
                        "Confidence": f"{confidence_emoji} {confidence_pct:.1f}%",
                        "Score": f"{clause['confidence']:.4f}"
                    }
                    
                    # Add matched text if available
                    if clause.get('matched_text') and advanced_features:
                        matched_text = clause['matched_text'][:100] + "..." if len(clause['matched_text']) > 100 else clause['matched_text']
                        clause_info["Matched Text"] = matched_text
                    
                    clause_data.append(clause_info)
                
                # Display as dataframe
                clause_df = pd.DataFrame(clause_data)
                st.dataframe(clause_df, use_container_width=True, hide_index=True)
                
                # Enhanced visualizations
                st.subheader("📈 Visual Analytics")
                
                tab1, tab2, tab3 = st.tabs(["🎯 Confidence Distribution", "📊 Clause Categories", "🔍 Document Analysis"])
                
                with tab1:
                    # Enhanced confidence chart
                    fig = px.bar(
                        x=[c['confidence'] for c in results['detected_clauses']],
                        y=[c['clean_name'] for c in results['detected_clauses']],
                        orientation='h',
                        title='Clause Detection Confidence Scores',
                        labels={'x': 'Confidence Score', 'y': 'Clause Type'},
                        color=[c['confidence'] for c in results['detected_clauses']],
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(height=max(400, len(results['detected_clauses']) * 30))
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Clause category analysis
                    confidence_ranges = {
                        'High (≥0.7)': sum(1 for c in results['detected_clauses'] if c['confidence'] >= 0.7),
                        'Medium (0.3-0.7)': sum(1 for c in results['detected_clauses'] if 0.3 <= c['confidence'] < 0.7),
                        'Low (<0.3)': sum(1 for c in results['detected_clauses'] if c['confidence'] < 0.3)
                    }
                    
                    if any(confidence_ranges.values()):
                        fig_pie = px.pie(
                            values=list(confidence_ranges.values()),
                            names=list(confidence_ranges.keys()),
                            title='Confidence Distribution',
                            color_discrete_sequence=['#00CC96', '#FFA15A', '#EF553B']
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                with tab3:
                    # Document complexity visualization
                    complexity_metrics = {
                        'Word Count': complexity.get('word_count', 0),
                        'Sentence Count': complexity.get('sentence_count', 0),
                        'Legal Terms': complexity.get('legal_term_count', 0),
                        'Structure Indicators': complexity.get('structure_indicators', 0)
                    }
                    
                    fig_complexity = px.bar(
                        x=list(complexity_metrics.keys()),
                        y=list(complexity_metrics.values()),
                        title='Document Complexity Metrics',
                        color=list(complexity_metrics.values()),
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_complexity, use_container_width=True)
                    
                    # Additional complexity info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Legal Term Density", f"{complexity.get('legal_term_density', 0):.1f}%")
                        st.metric("Avg Sentence Length", f"{complexity.get('avg_sentence_length', 0):.1f} words")
                    with col2:
                        st.metric("Complexity Score", f"{complexity.get('complexity_score', 0):.2f}")
                        st.metric("Readability Score", f"{complexity.get('readability_score', 0):.1f}/100")
            
            else:
                st.info("💡 No clauses detected above the specified confidence threshold. Try:")
                st.markdown("""
                - Lowering the confidence threshold
                - Checking if the text contains legal clauses
                - Using different sample text
                - Verifying the document format
                """)
            
            # Processing details
            with st.expander("🔧 Processing Details"):
                proc_info = results.get('processing_info', {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.json({k: v for k, v in proc_info.items() if k != 'config_used'})
                
                with col2:
                    if 'config_used' in proc_info:
                        st.write("**Configuration Used:**")
                        st.json(proc_info['config_used'])
            
            # Export section
            st.subheader("💾 Export Results")
            
            # Prepare comprehensive export data
            export_data = {
                'analysis_metadata': {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'extractor_version': '2.0.0',
                    'model_info': extractor.get_model_info()
                },
                'document_analysis': results.get('complexity_analysis', {}),
                'statistics': results.get('statistics', {}),
                'detected_clauses': results['detected_clauses'],
                'processing_info': results.get('processing_info', {})
            }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📄 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"clause_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                if results['detected_clauses']:
                    csv_data = pd.DataFrame(results['detected_clauses']).to_csv(index=False)
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv_data,
                        file_name=f"detected_clauses_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col3:
                # Create detailed report
                report_lines = [
                    f"# Legal Clause Analysis Report",
                    f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"",
                    f"## Summary",
                    f"- Total clauses detected: {results['clause_count']}",
                    f"- Average confidence: {results.get('statistics', {}).get('avg_confidence', 0):.3f}",
                    f"- Document complexity: {results.get('complexity_analysis', {}).get('complexity_level', 'Unknown')}",
                    f"",
                    f"## Detected Clauses"
                ]
                
                for i, clause in enumerate(results['detected_clauses'], 1):
                    report_lines.append(f"{i}. {clause['clean_name']}: {clause['confidence']:.3f}")
                
                report_text = "\n".join(report_lines)
                
                st.download_button(
                    label="📝 Download Report",
                    data=report_text,
                    file_name=f"clause_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error during analysis: {e}")
            logger.error(f"Analysis error: {e}")

# Legacy compatibility functions
def extract_clauses(text: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
    """Legacy function for backward compatibility"""
    extractor = load_clause_extractor()
    if extractor is None:
        return []
    
    config = ExtractionConfig(confidence_threshold=threshold)
    results = extractor.extract_clauses(text, config)
    return results.get('detected_clauses', [])

def render_clause_extraction_interface():
    """Legacy interface function - redirects to enhanced version"""
    render_enhanced_clause_interface()

# Main execution for testing
if __name__ == "__main__":
    # Test the enhanced extractor
    print("Testing Enhanced Legal Clause Extractor...")
    
    extractor = LegalClauseExtractor()
    
    sample_text = """
    This Software License Agreement ("Agreement") is effective as of January 1, 2024 
    between TechCorp Inc., a Delaware corporation ("Licensor"), and ClientCorp LLC 
    ("Licensee"). The term of this Agreement shall be three (3) years unless terminated 
    earlier in accordance with the provisions hereof. This Agreement shall be governed 
    by the laws of the State of California. Either party may terminate this Agreement 
    for convenience upon thirty (30) days written notice. The Licensee shall pay 
    liquidated damages of $50,000 for any breach of this Agreement.
    """
    
    config = ExtractionConfig(
        confidence_threshold=0.2,
        return_matched_text=True,
        return_positions=True
    )
    
    results = extractor.extract_clauses(sample_text, config)
    
    print(f"\nDetected {results['clause_count']} clauses:")
    for clause in results['detected_clauses']:
        print(f"- {clause['clean_name']}: {clause['confidence']:.3f}")
        if clause.get('matched_text'):
            print(f"  Text: {clause['matched_text'][:100]}...")
    
    print(f"\nModel Info:")
    model_info = extractor.get_model_info()
    print(f"- Model loaded: {model_info['model_loaded']}")
    print(f"- Device: {model_info['device']}")
    print(f"- Clause types: {model_info['num_clause_types']}")