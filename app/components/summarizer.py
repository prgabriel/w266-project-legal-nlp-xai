"""
Legal Document Summarization Component for Streamlit App
Enhanced T5-based abstractive summarization for legal contracts with domain optimization
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
import time
import logging
from functools import lru_cache
import warnings

# NLP and summarization libraries
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration
)
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

# Streamlit and visualization
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from scripts.utils import load_data, clean_clause_name, PROJECT, preprocess_text
    from scripts.evaluation_metrics import LegalNLPEvaluator
except ImportError as e:
    logger.warning(f"Could not import some utilities: {e}")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

@dataclass
class SummarizationConfig:
    """Configuration for legal document summarization"""
    model_name: str = 't5-base'
    max_input_length: int = 1024
    max_output_length: int = 256
    min_output_length: int = 50
    num_beams: int = 4
    do_sample: bool = False
    temperature: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
    summary_type: str = 'extractive_abstractive'  # 'extractive', 'abstractive', 'extractive_abstractive'
    focus_clauses: Optional[List[str]] = None
    include_key_terms: bool = True
    preserve_legal_structure: bool = True

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.min_output_length >= self.max_output_length:
            raise ValueError("min_output_length must be less than max_output_length")
        
        if self.max_input_length <= 0:
            raise ValueError("max_input_length must be positive")
        
        if self.summary_type not in ['extractive', 'abstractive', 'extractive_abstractive']:
            raise ValueError(f"Invalid summary_type: {self.summary_type}")

@dataclass
class SummaryResult:
    """Result of legal document summarization"""
    original_text: str
    summary: str
    summary_type: str
    length_compression_ratio: float
    key_sentences: List[str]
    key_terms: List[str]
    legal_entities: Dict[str, List[str]]
    readability_score: float
    rouge_scores: Optional[Dict[str, float]] = None
    processing_time: float = 0.0
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class LegalDocumentSummarizer:
    """
    Enhanced Legal Document Summarization Framework
    Supports T5-based abstractive summarization with legal domain optimization
    """
    
    # Legal-specific terms and phrases
    LEGAL_TERMS = [
        'agreement', 'contract', 'party', 'parties', 'clause', 'provision',
        'terms', 'conditions', 'whereas', 'therefore', 'notwithstanding',
        'pursuant', 'covenant', 'warranty', 'representation', 'indemnification',
        'liability', 'damages', 'breach', 'default', 'termination', 'expiration',
        'governing law', 'jurisdiction', 'arbitration', 'force majeure'
    ]
    
    # Legal entity patterns
    LEGAL_ENTITY_PATTERNS = {
        'dates': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
        'money': r'\$[\d,]+(?:\.\d{2})?|\b\d+\s*(?:dollars?|USD|million|billion)\b',
        'percentages': r'\b\d+(?:\.\d+)?%|\b\d+\s*percent\b',
        'time_periods': r'\b\d+\s*(?:days?|weeks?|months?|years?)\b',
        'companies': r'\b[A-Z][a-zA-Z\s&,\.]*(?:Inc\.|LLC|Corp\.|Corporation|Company|Co\.|Ltd\.)\b'
    }
    
    def __init__(self, 
                 model_path: str = None,
                 config: Optional[SummarizationConfig] = None,
                 cache_dir: str = None):
        """
        Initialize the Legal Document Summarizer
        
        Args:
            model_path: Path to trained T5 model directory
            config: Configuration for summarization parameters
            cache_dir: Directory for caching models and data
        """
        self.config = config or SummarizationConfig()
        self.model = None
        self.tokenizer = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Path management
        self.project_root = Path(__file__).parent.parent.parent
        self.model_path = Path(model_path) if model_path else self.project_root / 'models' / 't5'
        self.cache_dir = Path(cache_dir) if cache_dir else self.project_root / 'app' / '.cache'
        
        # Performance tracking
        self._performance_metrics = {}
        self._model_loaded = False
        
        # Initialize model and components
        self._initialize()
    
    def _initialize(self):
        """Initialize model, tokenizer, and supporting components"""
        try:
            start_time = time.time()
            
            # Load model and tokenizer
            self._load_model()
            
            # Initialize supporting components
            self._initialize_legal_preprocessing()
            
            init_time = time.time() - start_time
            self._performance_metrics['initialization_time'] = init_time
            logger.info(f"Legal summarizer initialized in {init_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            self._load_fallback()
    
    def _load_model(self):
        """Load T5 model with fallback options"""
        try:
            # Try loading fine-tuned model first
            if self.model_path.exists():
                self._load_fine_tuned_model()
            else:
                logger.warning(f"Fine-tuned model not found at {self.model_path}")
                self._load_default_model()
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._load_default_model()
    
    def _load_fine_tuned_model(self):
        """Load fine-tuned T5 model for legal documents"""
        try:
            # Load model configuration if available
            config_path = self.model_path / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                    self.config.model_name = model_config.get('model_name', self.config.model_name)
            
            # Load model and tokenizer
            self.model = T5ForConditionalGeneration.from_pretrained(str(self.model_path))
            
            tokenizer_path = self.model_path / 'tokenizer'
            if tokenizer_path.exists():
                self.tokenizer = T5Tokenizer.from_pretrained(str(tokenizer_path))
            else:
                self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
            
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
            
            logger.info(f"Loaded fine-tuned T5 model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            self._load_default_model()
    
    def _load_default_model(self):
        """Load default T5 model with better error handling"""
        try:
            logger.info(f"Loading default T5 model: {self.config.model_name}")
            
            # Use t5-small if t5-base fails (smaller, more reliable)
            fallback_models = [self.config.model_name, 't5-small', 't5-base']
            
            for model_name in fallback_models:
                try:
                    logger.info(f"Attempting to load {model_name}...")
                    self.model = T5ForConditionalGeneration.from_pretrained(model_name)
                    self.tokenizer = T5Tokenizer.from_pretrained(model_name)
                    
                    # Add legal tokens after loading
                    self._add_legal_tokens()
                    
                    self.model.to(self.device)
                    self.model.eval()
                    self._model_loaded = True
                    
                    logger.info(f"Successfully loaded T5 model: {model_name}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            else:
                # If all models fail, use fallback
                raise Exception("All T5 models failed to load")
                
        except Exception as e:
            logger.error(f"Critical error loading any T5 model: {e}")
            self._load_fallback()
    
    def _load_fallback(self):
        """Enhanced fallback with extractive summarization"""
        logger.warning("Loading fallback mode - extractive summarization only")
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        
        # Initialize basic extractive summarization
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self._extractive_ready = True
            logger.info("Extractive summarization fallback ready")
        except ImportError:
            logger.error("Extractive summarization fallback not available")
            self._extractive_ready = False

    def _initialize_legal_preprocessing(self):
        """Initialize legal text preprocessing components"""
        try:
            # Compile regex patterns for efficiency
            self._entity_patterns = {
                name: re.compile(pattern, re.IGNORECASE)
                for name, pattern in self.LEGAL_ENTITY_PATTERNS.items()
            }
            
            # Create legal term lookup
            self._legal_terms_set = set(term.lower() for term in self.LEGAL_TERMS)
            
        except Exception as e:
            logger.warning(f"Error initializing preprocessing: {e}")
    
    def _add_legal_tokens(self):
        """Add legal-specific tokens to T5 tokenizer"""
        if not self.tokenizer:
            logger.warning("Tokenizer not initialized, skipping legal token addition")
            return
        
        legal_tokens = [
            '<legal_clause>', '<contract_section>', '<party_name>',
            '<date_reference>', '<monetary_amount>', '<legal_term>'
        ]
        
        # For T5 tokenizer, check existing vocabulary using get_vocab()
        try:
            existing_vocab = self.tokenizer.get_vocab()  # Use get_vocab() instead of .vocab
            new_tokens = [token for token in legal_tokens if token not in existing_vocab]
            
            if new_tokens:
                # Add new tokens
                num_added = self.tokenizer.add_tokens(new_tokens)
                
                if num_added > 0 and self.model:
                    # Resize model embeddings to accommodate new tokens
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    logger.info(f"Added {num_added} legal tokens to T5 tokenizer: {new_tokens}")
            else:
                logger.info("All legal tokens already exist in tokenizer vocabulary")
                
        except Exception as e:
            logger.error(f"Error adding legal tokens: {e}")
            # Fallback: try to add tokens without checking vocabulary
            try:
                num_added = self.tokenizer.add_tokens(legal_tokens)
                if num_added > 0 and self.model:
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    logger.info(f"Added {num_added} legal tokens (fallback method)")
            except Exception as fallback_error:
                logger.error(f"Fallback token addition also failed: {fallback_error}")

    @lru_cache(maxsize=64)
    def _preprocess_legal_text(self, text: str) -> str:
        """Cached legal text preprocessing"""
        return self._preprocess_legal_text_impl(text)
    
    def _preprocess_legal_text_impl(self, text: str) -> str:
        """Enhanced legal text preprocessing"""
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
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
        
        # Normalize legal citations and references
        text = re.sub(r'\bSection\s+(\d+(?:\.\d+)?)\b', r'Section \1', text)
        text = re.sub(r'\bArticle\s+(\d+(?:\.\d+)?)\b', r'Article \1', text)
        
        return text

    def summarize_document(self, 
                          text: str,
                          config: Optional[SummarizationConfig] = None) -> SummaryResult:
        """
        Generate comprehensive summary of legal document
        
        Args:
            text: Input legal document text
            config: Configuration override for this summarization
            
        Returns:
            SummaryResult with comprehensive summary information
        """
        summarization_config = config or self.config
        start_time = time.time()
        
        try:
            # Preprocess text
            processed_text = self._preprocess_legal_text(text)
            
            # Extract key information
            key_sentences = self._extract_key_sentences(processed_text)
            key_terms = self._extract_key_terms(processed_text)
            legal_entities = self._extract_legal_entities(processed_text)
            
            # Generate summary based on configuration
            if summarization_config.summary_type == 'extractive':
                summary = self._generate_extractive_summary(processed_text, key_sentences, summarization_config)
            elif summarization_config.summary_type == 'abstractive':
                summary = self._generate_abstractive_summary(processed_text, summarization_config)
            else:  # extractive_abstractive
                summary = self._generate_hybrid_summary(processed_text, key_sentences, summarization_config)
            
            # Calculate metrics
            compression_ratio = len(summary) / len(text) if text else 0
            readability_score = self._calculate_readability_score(summary)
            confidence_score = self._calculate_confidence_score(text, summary)
            
            # Calculate ROUGE scores if possible
            rouge_scores = None
            if len(key_sentences) > 0:
                reference_summary = ' '.join(key_sentences[:3])  # Use top sentences as reference
                rouge_scores = self._calculate_rouge_scores(summary, reference_summary)
            
            processing_time = time.time() - start_time
            
            return SummaryResult(
                original_text=text,
                summary=summary,
                summary_type=summarization_config.summary_type,
                length_compression_ratio=compression_ratio,
                key_sentences=key_sentences,
                key_terms=key_terms,
                legal_entities=legal_entities,
                readability_score=readability_score,
                rouge_scores=rouge_scores,
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return self._create_fallback_summary(text, str(e))

    def _extract_key_sentences(self, text: str) -> List[str]:
        """Extract key sentences using importance scoring"""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= 3:
                return sentences
            
            # Initialize preprocessing components if needed
            if not hasattr(self, '_legal_terms_set'):
                self._initialize_legal_preprocessing()
            
            # Score sentences based on legal importance
            sentence_scores = []
            
            for sentence in sentences:
                score = 0
                sentence_lower = sentence.lower()
                
                # Legal term presence
                legal_term_count = sum(1 for term in self._legal_terms_set if term in sentence_lower)
                score += legal_term_count * 2
                
                # Sentence position (first and last sentences are often important)
                position_weight = 1.5 if sentences.index(sentence) in [0, len(sentences)-1] else 1.0
                score *= position_weight
                
                # Sentence length (moderate length preferred)
                word_count = len(sentence.split())
                if 10 <= word_count <= 40:
                    score += 1
                
                # Contains numbers, dates, or monetary amounts
                if re.search(r'\d', sentence):
                    score += 1
                
                # Contains legal entities
                if hasattr(self, '_entity_patterns'):
                    entity_count = sum(1 for pattern in self._entity_patterns.values() 
                                     if pattern.search(sentence))
                    score += entity_count
                
                sentence_scores.append((sentence, score))
            
            # Sort by score and return top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [sent for sent, score in sentence_scores[:min(10, len(sentences)//2)]]
            
            # Maintain original order
            ordered_sentences = []
            for sentence in sentences:
                if sentence in top_sentences:
                    ordered_sentences.append(sentence)
            
            return ordered_sentences
            
        except Exception as e:
            logger.error(f"Error extracting key sentences: {e}")
            return sent_tokenize(text)[:5]  # Fallback to first 5 sentences

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key legal terms and phrases"""
        try:
            # Initialize if needed
            if not hasattr(self, '_legal_terms_set'):
                self._initialize_legal_preprocessing()
                
            # Tokenize and find legal terms
            words = word_tokenize(text.lower())
            
            # Find legal terms
            found_terms = []
            for word in words:
                if word in self._legal_terms_set:
                    found_terms.append(word)
            
            # Find legal phrases (2-3 word combinations)
            legal_phrases = [
                'governing law', 'force majeure', 'intellectual property',
                'confidential information', 'material breach', 'liquidated damages',
                'indemnification clause', 'termination clause', 'renewal term'
            ]
            
            text_lower = text.lower()
            found_phrases = [phrase for phrase in legal_phrases if phrase in text_lower]
            
            # Combine and deduplicate
            all_terms = list(set(found_terms + found_phrases))
            
            # Sort by frequency in text
            term_freq = [(term, text_lower.count(term)) for term in all_terms]
            term_freq.sort(key=lambda x: x[1], reverse=True)
            
            return [term for term, freq in term_freq[:20]]
            
        except Exception as e:
            logger.error(f"Error extracting key terms: {e}")
            return []

    def _extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities using regex patterns"""
        try:
            entities = {}
            
            if hasattr(self, '_entity_patterns'):
                for entity_type, pattern in self._entity_patterns.items():
                    matches = pattern.findall(text)
                    entities[entity_type] = list(set(matches))  # Remove duplicates
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting legal entities: {e}")
            return {}

    def _generate_extractive_summary(self, 
                                   text: str, 
                                   key_sentences: List[str],
                                   config: SummarizationConfig) -> str:
        """Generate extractive summary using key sentences"""
        if not key_sentences:
            sentences = sent_tokenize(text)
            key_sentences = sentences[:3]
        
        # Select sentences to meet length requirements
        summary_sentences = []
        current_length = 0
        
        for sentence in key_sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= config.max_output_length:
                summary_sentences.append(sentence)
                current_length += sentence_length
            else:
                break
        
        # Ensure minimum length
        if current_length < config.min_output_length and len(summary_sentences) < len(key_sentences):
            remaining_sentences = key_sentences[len(summary_sentences):]
            for sentence in remaining_sentences:
                summary_sentences.append(sentence)
                current_length += len(sentence.split())
                if current_length >= config.min_output_length:
                    break
        
        return ' '.join(summary_sentences)

    def _generate_abstractive_summary(self, 
                                    text: str,
                                    config: SummarizationConfig) -> str:
        """Generate abstractive summary using T5 model"""
        try:
            if not self._model_loaded:
                logger.warning("Model not loaded, falling back to extractive summary")
                return self._generate_extractive_summary(text, self._extract_key_sentences(text), config)
            
            # Prepare input with task prefix
            input_text = f"summarize legal document: {text}"
            
            # Tokenize
            encoding = self.tokenizer(
                input_text,
                max_length=config.max_input_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding.input_ids.to(self.device)
            attention_mask = encoding.attention_mask.to(self.device)
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=config.max_output_length,
                    min_length=config.min_output_length,
                    num_beams=config.num_beams,
                    do_sample=config.do_sample,
                    temperature=config.temperature,
                    length_penalty=config.length_penalty,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    early_stopping=config.early_stopping
                )
            
            # Decode summary
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process summary
            summary = self._post_process_summary(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in abstractive summarization: {e}")
            return self._generate_extractive_summary(text, self._extract_key_sentences(text), config)

    def _generate_hybrid_summary(self, 
                                text: str,
                                key_sentences: List[str],
                                config: SummarizationConfig) -> str:
        """Generate hybrid extractive-abstractive summary"""
        try:
            # Generate both types of summaries
            extractive_summary = self._generate_extractive_summary(text, key_sentences, config)
            
            if self._model_loaded:
                # Use extractive summary as input for abstractive refinement
                abstractive_config = SummarizationConfig(
                    max_input_length=512,
                    max_output_length=config.max_output_length,
                    min_output_length=config.min_output_length
                )
                abstractive_summary = self._generate_abstractive_summary(extractive_summary, abstractive_config)
                
                # Combine insights from both approaches
                if len(abstractive_summary.split()) >= config.min_output_length:
                    return abstractive_summary
            
            return extractive_summary
            
        except Exception as e:
            logger.error(f"Error in hybrid summarization: {e}")
            return self._generate_extractive_summary(text, key_sentences, config)

    def _post_process_summary(self, summary: str) -> str:
        """Post-process generated summary for legal domain"""
        summary = summary.strip()
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        
        if summary and summary[-1] not in '.!?':
            summary += '.'
        
        # Legal term capitalization
        legal_term_fixes = {
            r'\bagreement\b': 'Agreement',
            r'\bcontract\b': 'Contract',
            r'\bparty\b': 'Party',
            r'\bparties\b': 'Parties'
        }
        
        for pattern, replacement in legal_term_fixes.items():
            summary = re.sub(pattern, replacement, summary, flags=re.IGNORECASE)
        
        return summary

    def _calculate_readability_score(self, text: str) -> float:
        """Calculate readability score (simplified Flesch score)"""
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            if not sentences or not words:
                return 0.0
            
            # More explicit protection
            sentence_count = max(len(sentences), 1)
            word_count = max(len(words), 1)
            
            avg_sentence_length = word_count / sentence_count
            
            # Count syllables (simplified)
            syllable_count = 0
            for word in words:
                syllables = max(1, len([char for char in word.lower() if char in 'aeiouy']))
                syllable_count += syllables
            
            avg_syllables_per_word = syllable_count / len(words)
            
            # Simplified Flesch reading ease score
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Normalize to 0-100 scale
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return 50.0  # Default middle score

    def _calculate_confidence_score(self, original: str, summary: str) -> float:
        """Calculate confidence score for the summary"""
        try:
            # Factors that contribute to confidence
            score = 0.0
            
            # Length appropriateness
            compression_ratio = len(summary) / len(original) if original else 0
            if 0.1 <= compression_ratio <= 0.3:
                score += 0.3
            
            # Key term preservation
            original_terms = set(self._extract_key_terms(original))
            summary_terms = set(self._extract_key_terms(summary))
            
            if original_terms:
                preservation_ratio = len(summary_terms.intersection(original_terms)) / len(original_terms)
                score += preservation_ratio * 0.4
            
            # Coherence (simplified)
            summary_sentences = sent_tokenize(summary)
            if len(summary_sentences) >= 2:
                score += 0.2
            
            # Grammar and structure (simplified check)
            if summary.strip() and summary[0].isupper() and summary.strip()[-1] in '.!?':
                score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, generated)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def _create_fallback_summary(self, text: str, error_msg: str) -> SummaryResult:
        """Create fallback summary when main process fails"""
        # Simple extractive fallback
        sentences = sent_tokenize(text)
        summary = ' '.join(sentences[:3]) if sentences else "Unable to generate summary."
        
        return SummaryResult(
            original_text=text,
            summary=summary,
            summary_type='fallback_extractive',
            length_compression_ratio=len(summary) / len(text) if text else 0,
            key_sentences=sentences[:5],
            key_terms=[],
            legal_entities={},
            readability_score=50.0,
            processing_time=0.0,
            confidence_score=0.2
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_loaded': self._model_loaded,
            'model_name': self.config.model_name,
            'device': str(self.device),
            'model_path': str(self.model_path),
            'performance_metrics': self._performance_metrics,
            'config': asdict(self.config),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }


def summarize_text(text: str, 
                   summary_type: str = "extractive_abstractive",
                   max_length: int = 256,
                   min_length: int = 50) -> Dict[str, Any]:
    """
    Simple function to summarize text - wrapper for the main summarizer
    
    Args:
        text: Input text to summarize
        summary_type: Type of summarization ("extractive", "abstractive", "extractive_abstractive")
        max_length: Maximum length of summary
        min_length: Minimum length of summary
        
    Returns:
        Dictionary with summary results
    """
    try:
        # Create configuration
        config = SummarizationConfig(
            summary_type=summary_type,
            max_output_length=max_length,
            min_output_length=min_length
        )
        
        # Initialize summarizer (cached)
        @st.cache_resource
        def get_summarizer():
            return LegalDocumentSummarizer(config=config)
        
        summarizer = get_summarizer()
        
        # Generate summary
        result = summarizer.summarize_document(text, config)
        
        # Return as dictionary
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Error in summarize_text: {e}")
        return {
            'summary': f"Error generating summary: {str(e)}",
            'summary_type': 'error',
            'length_compression_ratio': 0.0,
            'key_sentences': [],
            'key_terms': [],
            'legal_entities': {},
            'readability_score': 0.0,
            'processing_time': 0.0,
            'confidence_score': 0.0
        }


# Streamlit Interface Function
def render_enhanced_summarization_interface():
    """Render the enhanced summarization interface for Streamlit"""
    st.title("🏛️ Legal Document Summarization")
    st.markdown("Enhanced T5-based abstractive summarization for legal contracts and clauses")
    
    # Initialize summarizer
    @st.cache_resource
    def load_summarizer():
        return LegalDocumentSummarizer()
    
    try:
        summarizer = load_summarizer()
        
        # Configuration sidebar
        st.sidebar.header("Summarization Settings")
        
        summary_type = st.sidebar.selectbox(
            "Summary Type",
            ["extractive_abstractive", "extractive", "abstractive"],
            help="Choose the type of summarization"
        )
        
        max_length = st.sidebar.slider("Max Summary Length", 50, 500, 256)
        min_length = st.sidebar.slider("Min Summary Length", 20, 200, 50)
        
        # Model Information in sidebar (like clause extraction)
        st.sidebar.header("Model Information")
        model_info = summarizer.get_model_info()
        
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
        st.sidebar.write("**Model Name**")
        st.sidebar.write(model_info['model_name'])
        
        # Initialization Time
        if 'performance_metrics' in model_info and 'initialization_time' in model_info['performance_metrics']:
            st.sidebar.write("**Init Time**")
            st.sidebar.write(f"{model_info['performance_metrics']['initialization_time']:.2f}s")
        
        # Technical Details (collapsible)
        with st.sidebar.expander("Technical Details"):
            st.json({
                'torch_version': model_info.get('torch_version', 'Unknown'),
                'cuda_available': model_info.get('cuda_available', False),
                'model_path': model_info.get('model_path', 'Default'),
                'config': model_info.get('config', {})
            })
        
        # Text input
        st.header("Document Input")
        input_text = st.text_area(
            "Enter legal document text:",
            height=300,
            placeholder="Paste your legal document text here..."
        )
        
        if st.button("Generate Summary", type="primary"):
            if input_text.strip():
                with st.spinner("Generating summary..."):
                    # Create configuration
                    config = SummarizationConfig(
                        summary_type=summary_type,
                        max_output_length=max_length,
                        min_output_length=min_length
                    )
                    
                    # Generate summary
                    result = summarizer.summarize_document(input_text, config)
                    
                    # Display results
                    st.header("Summary Results")
                    
                    # Main summary
                    st.subheader("Generated Summary")
                    st.write(result.summary)
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Compression Ratio", f"{result.length_compression_ratio:.2%}")
                    with col2:
                        st.metric("Confidence Score", f"{result.confidence_score:.2%}")
                    with col3:
                        st.metric("Readability Score", f"{result.readability_score:.1f}")
                    with col4:
                        st.metric("Processing Time", f"{result.processing_time:.2f}s")
                    
                    # Additional information
                    if result.key_terms:
                        st.subheader("Key Legal Terms")
                        st.write(", ".join(result.key_terms[:10]))
                    
                    if result.legal_entities:
                        st.subheader("Extracted Entities")
                        for entity_type, entities in result.legal_entities.items():
                            if entities:
                                st.write(f"**{entity_type.title()}:** {', '.join(entities[:5])}")
                    
                    # ROUGE scores if available
                    if result.rouge_scores:
                        st.subheader("ROUGE Scores")
                        rouge_col1, rouge_col2, rouge_col3 = st.columns(3)
                        with rouge_col1:
                            st.metric("ROUGE-1", f"{result.rouge_scores['rouge1']:.3f}")
                        with rouge_col2:
                            st.metric("ROUGE-2", f"{result.rouge_scores['rouge2']:.3f}")
                        with rouge_col3:
                            st.metric("ROUGE-L", f"{result.rouge_scores['rougeL']:.3f}")
            else:
                st.warning("Please enter some text to summarize.")
    
    except Exception as e:
        st.error(f"Error initializing summarizer: {e}")
        st.info("Please check that the required models are available.")


if __name__ == "__main__":
    render_enhanced_summarization_interface()