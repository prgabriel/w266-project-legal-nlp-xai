"""
Legal NLP Explainability Component for Streamlit App
Comprehensive explainability with SHAP, LIME, and attention analysis for legal clause detection
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from functools import lru_cache
import warnings

# Explainability libraries
import shap
import lime
from lime.lime_text import LimeTextExplainer

# Updated Captum imports - fix for TokenReferenceGenerator
try:
    from captum.attr import IntegratedGradients, LayerIntegratedGradients
    from captum.attr import TokenReferenceGenerator
except ImportError:
    # Fallback for newer versions of Captum
    from captum.attr import IntegratedGradients, LayerIntegratedGradients
    try:
        from captum.attr._utils.token_reference import TokenReferenceGenerator
    except ImportError:
        # Manual implementation if not available
        class TokenReferenceGenerator:
            def __init__(self, reference_token_idx=0):
                self.reference_token_idx = reference_token_idx
            
            def generate_reference(self, sequence_length, device=None):
                """Generate reference tensor filled with reference token"""
                reference = torch.full((sequence_length,), self.reference_token_idx, dtype=torch.long)
                if device:
                    reference = reference.to(device)
                return reference

try:
    from captum.attr import visualization as viz
except ImportError:
    viz = None
    print("Captum visualization not available")

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.utils import load_data, clean_clause_name, PROJECT
from scripts.evaluation_metrics import LegalNLPEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class ExplanationConfig:
    """Configuration for explainability analysis"""
    explanation_method: str = 'shap'  # 'shap', 'lime', 'integrated_gradients', 'attention'
    top_k_features: int = 20
    max_tokens_display: int = 100
    include_negative_attributions: bool = True
    normalize_attributions: bool = True
    aggregate_method: str = 'mean'  # 'mean', 'max', 'sum'
    confidence_threshold: float = 0.3
    generate_plots: bool = True
    save_explanations: bool = False

@dataclass
class ExplanationResult:
    """Result of explainability analysis"""
    method: str
    clause_type: str
    confidence: float
    feature_attributions: Dict[str, float]
    top_positive_features: List[Tuple[str, float]]
    top_negative_features: List[Tuple[str, float]]
    explanation_summary: str
    visualization_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class LegalExplainer:
    """
    Comprehensive Legal NLP Explainability Framework
    Supports SHAP, LIME, Integrated Gradients, and attention analysis
    """
    
    def __init__(self, 
                 model=None, 
                 tokenizer=None, 
                 clause_types: Optional[List[str]] = None,
                 clean_clause_names: Optional[Dict[str, str]] = None,
                 config: Optional[ExplanationConfig] = None):
        """
        Initialize the Legal Explainer
        
        Args:
            model: The trained model for explanation
            tokenizer: Tokenizer associated with the model
            clause_types: List of legal clause types
            clean_clause_names: Mapping of clause types to clean names
            config: Configuration for explanation methods
        """
        self.model = model
        self.tokenizer = tokenizer
        self.clause_types = clause_types or []
        self.clean_clause_names = clean_clause_names or {}
        self.config = config or ExplanationConfig()
        
        # Initialize explainer objects
        self.shap_explainer = None
        self.lime_explainer = None
        self.ig_explainer = None
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Performance tracking
        self._explanation_cache = {}
        self._performance_metrics = {}
        
        # Initialize explainers
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize explainability frameworks"""
        try:
            if self.model is not None and self.tokenizer is not None:
                self._initialize_shap()
                self._initialize_lime()
                self._initialize_integrated_gradients()
                logger.info("Explainability frameworks initialized successfully")
            else:
                logger.warning("Model or tokenizer not provided - explainers not initialized")
        except Exception as e:
            logger.error(f"Error initializing explainers: {e}")
    
    def _initialize_shap(self):
        """Initialize SHAP explainer"""
        try:
            # Create a wrapper function for SHAP
            def model_wrapper(texts):
                """Wrapper function for SHAP to handle text input"""
                if isinstance(texts, np.ndarray):
                    texts = texts.tolist()
                
                predictions = []
                for text in texts:
                    if isinstance(text, (list, np.ndarray)):
                        # Handle tokenized input
                        input_ids = torch.tensor(text).unsqueeze(0).to(self.device)
                        attention_mask = (input_ids != 0).float()
                    else:
                        # Handle raw text
                        encoding = self.tokenizer(
                            text,
                            truncation=True,
                            padding='max_length',
                            max_length=512,
                            return_tensors='pt'
                        )
                        input_ids = encoding['input_ids'].to(self.device)
                        attention_mask = encoding['attention_mask'].to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(input_ids, attention_mask)
                        probabilities = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
                        predictions.append(probabilities)
                
                return np.array(predictions)
            
            # Initialize SHAP explainer with the wrapper
            self.model_wrapper = model_wrapper
            self.shap_explainer = shap.Explainer(model_wrapper, self.tokenizer)
            
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _initialize_lime(self):
        """Initialize LIME explainer"""
        try:
            def lime_predict_fn(texts):
                """Prediction function for LIME"""
                predictions = []
                for text in texts:
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=512,
                        return_tensors='pt'
                    )
                    
                    input_ids = encoding['input_ids'].to(self.device)
                    attention_mask = encoding['attention_mask'].to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(input_ids, attention_mask)
                        probabilities = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
                        predictions.append(probabilities)
                
                return np.array(predictions)
            
            self.lime_explainer = LimeTextExplainer(
                class_names=[self.clean_clause_names.get(ct, ct) for ct in self.clause_types],
                feature_selection='auto',
                training_data_stats=None,
                verbose=False
            )
            self.lime_predict_fn = lime_predict_fn
            
        except Exception as e:
            logger.warning(f"Failed to initialize LIME explainer: {e}")
            self.lime_explainer = None
    
    def _initialize_integrated_gradients(self):
        """Initialize Integrated Gradients explainer"""
        try:
            if hasattr(self.model, 'bert'):
                # For BERT-based models
                self.ig_explainer = IntegratedGradients(self.model)
                self.token_reference = TokenReferenceGenerator(
                    reference_token_idx=self.tokenizer.pad_token_id
                )
            else:
                logger.warning("Model structure not compatible with Integrated Gradients")
                self.ig_explainer = None
                
        except Exception as e:
            logger.warning(f"Failed to initialize Integrated Gradients: {e}")
            self.ig_explainer = None
    
    def explain_prediction(self, 
                          text: str, 
                          predicted_clauses: List[Dict[str, Any]],
                          config: Optional[ExplanationConfig] = None) -> Dict[str, ExplanationResult]:
        """
        Generate comprehensive explanations for legal clause predictions
        
        Args:
            text: Input legal text
            predicted_clauses: List of predicted clauses with confidence scores
            config: Configuration override for this explanation
            
        Returns:
            Dictionary mapping clause types to explanation results
        """
        explanation_config = config or self.config
        explanations = {}
        
        start_time = time.time()
        
        try:
            # Filter clauses above confidence threshold
            relevant_clauses = [
                clause for clause in predicted_clauses 
                if clause['confidence'] >= explanation_config.confidence_threshold
            ]
            
            if not relevant_clauses:
                logger.warning("No clauses above confidence threshold for explanation")
                return {}
            
            # Generate explanations for each relevant clause
            for clause in relevant_clauses:
                clause_type = clause['clause_type']
                clean_name = clause['clean_name']
                confidence = clause['confidence']
                
                try:
                    if explanation_config.explanation_method == 'shap':
                        explanation = self._explain_with_shap(text, clause_type, confidence)
                    elif explanation_config.explanation_method == 'lime':
                        explanation = self._explain_with_lime(text, clause_type, confidence)
                    elif explanation_config.explanation_method == 'integrated_gradients':
                        explanation = self._explain_with_integrated_gradients(text, clause_type, confidence)
                    elif explanation_config.explanation_method == 'attention':
                        explanation = self._explain_with_attention(text, clause_type, confidence)
                    else:
                        logger.warning(f"Unknown explanation method: {explanation_config.explanation_method}")
                        continue
                    
                    if explanation:
                        explanations[clean_name] = explanation
                        
                except Exception as e:
                    logger.error(f"Error explaining clause {clean_name}: {e}")
                    continue
            
            # Track performance
            explanation_time = time.time() - start_time
            self._performance_metrics['last_explanation_time'] = explanation_time
            self._performance_metrics['clauses_explained'] = len(explanations)
            
            logger.info(f"Generated explanations for {len(explanations)} clauses in {explanation_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in explanation generation: {e}")
        
        return explanations
    
    def _explain_with_shap(self, text: str, clause_type: str, confidence: float) -> Optional[ExplanationResult]:
        """Generate SHAP-based explanation"""
        try:
            if self.shap_explainer is None:
                logger.warning("SHAP explainer not initialized")
                return None
            
            # Tokenize text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            # Get SHAP values
            shap_values = self.shap_explainer([text])
            
            # Extract feature attributions
            tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
            
            # Get clause index
            clause_idx = self.clause_types.index(clause_type) if clause_type in self.clause_types else 0
            
            # Extract attributions for this clause
            if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
                attributions = shap_values.values[0, :, clause_idx]
            else:
                attributions = shap_values.values[0] if hasattr(shap_values, 'values') else shap_values[0]
            
            # Create feature attribution dictionary
            feature_attributions = {}
            for token, attribution in zip(tokens, attributions):
                if token not in ['[PAD]', '[CLS]', '[SEP]']:
                    feature_attributions[token] = float(attribution)
            
            # Sort features by attribution
            sorted_features = sorted(feature_attributions.items(), key=lambda x: abs(x[1]), reverse=True)
            top_positive = [(k, v) for k, v in sorted_features if v > 0][:self.config.top_k_features]
            top_negative = [(k, v) for k, v in sorted_features if v < 0][:self.config.top_k_features]
            
            # Generate summary
            summary = self._generate_explanation_summary('SHAP', clause_type, top_positive, top_negative)
            
            return ExplanationResult(
                method='SHAP',
                clause_type=clause_type,
                confidence=confidence,
                feature_attributions=feature_attributions,
                top_positive_features=top_positive,
                top_negative_features=top_negative,
                explanation_summary=summary,
                visualization_data={'shap_values': shap_values, 'tokens': tokens}
            )
            
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {e}")
            return None
    
    def _explain_with_lime(self, text: str, clause_type: str, confidence: float) -> Optional[ExplanationResult]:
        """Generate LIME-based explanation"""
        try:
            if self.lime_explainer is None:
                logger.warning("LIME explainer not initialized")
                return None
            
            # Get clause index
            clause_idx = self.clause_types.index(clause_type) if clause_type in self.clause_types else 0
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                text,
                self.lime_predict_fn,
                num_features=self.config.top_k_features,
                labels=[clause_idx]
            )
            
            # Extract feature attributions
            feature_attributions = {}
            lime_list = explanation.as_list(label=clause_idx)
            
            for feature, importance in lime_list:
                feature_attributions[feature] = importance
            
            # Sort features
            sorted_features = sorted(feature_attributions.items(), key=lambda x: abs(x[1]), reverse=True)
            top_positive = [(k, v) for k, v in sorted_features if v > 0]
            top_negative = [(k, v) for k, v in sorted_features if v < 0]
            
            # Generate summary
            summary = self._generate_explanation_summary('LIME', clause_type, top_positive, top_negative)
            
            return ExplanationResult(
                method='LIME',
                clause_type=clause_type,
                confidence=confidence,
                feature_attributions=feature_attributions,
                top_positive_features=top_positive,
                top_negative_features=top_negative,
                explanation_summary=summary,
                visualization_data={'lime_explanation': explanation}
            )
            
        except Exception as e:
            logger.error(f"Error in LIME explanation: {e}")
            return None
    
    def _explain_with_integrated_gradients(self, text: str, clause_type: str, confidence: float) -> Optional[ExplanationResult]:
        """Generate Integrated Gradients explanation"""
        try:
            if self.ig_explainer is None:
                logger.warning("Integrated Gradients explainer not initialized")
                return None
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Generate reference tokens
            reference_input_ids = self.token_reference.generate_reference(
                sequence_length=input_ids.shape[1],
                device=self.device
            ).unsqueeze(0)
            
            # Get clause index
            clause_idx = self.clause_types.index(clause_type) if clause_type in self.clause_types else 0
            
            # Calculate attributions
            attributions = self.ig_explainer.attribute(
                input_ids,
                reference_input_ids,
                target=clause_idx,
                additional_forward_args=(attention_mask,),
                n_steps=50
            )
            
            # Convert to feature attributions
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            attributions_np = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
            
            feature_attributions = {}
            for token, attribution in zip(tokens, attributions_np):
                if token not in ['[PAD]', '[CLS]', '[SEP]']:
                    feature_attributions[token] = float(attribution)
            
            # Sort features
            sorted_features = sorted(feature_attributions.items(), key=lambda x: abs(x[1]), reverse=True)
            top_positive = [(k, v) for k, v in sorted_features if v > 0][:self.config.top_k_features]
            top_negative = [(k, v) for k, v in sorted_features if v < 0][:self.config.top_k_features]
            
            # Generate summary
            summary = self._generate_explanation_summary('Integrated Gradients', clause_type, top_positive, top_negative)
            
            return ExplanationResult(
                method='Integrated Gradients',
                clause_type=clause_type,
                confidence=confidence,
                feature_attributions=feature_attributions,
                top_positive_features=top_positive,
                top_negative_features=top_negative,
                explanation_summary=summary,
                visualization_data={'attributions': attributions, 'tokens': tokens}
            )
            
        except Exception as e:
            logger.error(f"Error in Integrated Gradients explanation: {e}")
            return None
    
    def _explain_with_attention(self, text: str, clause_type: str, confidence: float) -> Optional[ExplanationResult]:
        """Generate attention-based explanation"""
        try:
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Get model output with attention weights
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, output_attentions=True)
            
            # Extract attention weights
            attention_weights = outputs.attentions if hasattr(outputs, 'attentions') else None
            
            if attention_weights is None:
                logger.warning("Model does not provide attention weights")
                return None
            
            # Average attention across layers and heads
            attention = torch.stack(attention_weights).mean(dim=0).mean(dim=1)  # [batch, seq_len, seq_len]
            attention = attention[0]  # Remove batch dimension
            
            # Get attention from [CLS] token to all other tokens
            cls_attention = attention[0, :].cpu().numpy()
            
            # Convert to feature attributions
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            feature_attributions = {}
            for token, attention_weight in zip(tokens, cls_attention):
                if token not in ['[PAD]', '[CLS]', '[SEP]']:
                    feature_attributions[token] = float(attention_weight)
            
            # Sort features by attention weight
            sorted_features = sorted(feature_attributions.items(), key=lambda x: x[1], reverse=True)
            top_positive = sorted_features[:self.config.top_k_features]
            top_negative = []  # Attention weights are typically positive
            
            # Generate summary
            summary = self._generate_explanation_summary('Attention', clause_type, top_positive, top_negative)
            
            return ExplanationResult(
                method='Attention',
                clause_type=clause_type,
                confidence=confidence,
                feature_attributions=feature_attributions,
                top_positive_features=top_positive,
                top_negative_features=top_negative,
                explanation_summary=summary,
                visualization_data={'attention_weights': attention_weights, 'tokens': tokens}
            )
            
        except Exception as e:
            logger.error(f"Error in attention explanation: {e}")
            return None
    
    def _generate_explanation_summary(self, 
                                    method: str, 
                                    clause_type: str, 
                                    top_positive: List[Tuple[str, float]], 
                                    top_negative: List[Tuple[str, float]]) -> str:
        """Generate a human-readable explanation summary"""
        clean_name = self.clean_clause_names.get(clause_type, clause_type)
        
        summary_parts = [
            f"**{method} Explanation for {clean_name}:**\n"
        ]
        
        if top_positive:
            summary_parts.append("**Key Supporting Evidence:**")
            for i, (feature, score) in enumerate(top_positive[:5], 1):
                summary_parts.append(f"{i}. '{feature}' (importance: {score:.3f})")
            summary_parts.append("")
        
        if top_negative and method != 'Attention':
            summary_parts.append("**Contradicting Evidence:**")
            for i, (feature, score) in enumerate(top_negative[:3], 1):
                summary_parts.append(f"{i}. '{feature}' (importance: {score:.3f})")
            summary_parts.append("")
        
        # Add interpretation
        if top_positive:
            key_terms = [term for term, _ in top_positive[:3]]
            summary_parts.append(
                f"The model's decision is primarily influenced by terms like: {', '.join(key_terms)}. "
                f"These words strongly indicate the presence of a {clean_name} clause."
            )
        
        return "\n".join(summary_parts)
    
    def create_explanation_visualizations(self, 
                                        explanations: Dict[str, ExplanationResult]) -> Dict[str, Any]:
        """Create comprehensive visualizations for explanations"""
        visualizations = {}
        
        try:
            # Feature importance heatmap
            if explanations:
                viz_data = self._create_feature_importance_heatmap(explanations)
                visualizations['feature_heatmap'] = viz_data
                
                # Clause comparison chart
                comparison_chart = self._create_clause_comparison_chart(explanations)
                visualizations['clause_comparison'] = comparison_chart
                
                # Attribution distribution
                attribution_dist = self._create_attribution_distribution(explanations)
                visualizations['attribution_distribution'] = attribution_dist
                
                # Word cloud data
                wordcloud_data = self._create_wordcloud_data(explanations)
                visualizations['wordcloud'] = wordcloud_data
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return visualizations
    
    def _create_feature_importance_heatmap(self, explanations: Dict[str, ExplanationResult]) -> Dict[str, Any]:
        """Create feature importance heatmap data"""
        try:
            # Collect all features and their importances
            feature_data = []
            
            for clause_name, explanation in explanations.items():
                for feature, importance in explanation.top_positive_features[:15]:
                    feature_data.append({
                        'clause': clause_name,
                        'feature': feature,
                        'importance': importance,
                        'type': 'positive'
                    })
                
                for feature, importance in explanation.top_negative_features[:10]:
                    feature_data.append({
                        'clause': clause_name,
                        'feature': feature,
                        'importance': abs(importance),  # Use absolute value for visualization
                        'type': 'negative'
                    })
            
            if not feature_data:
                return {}
            
            df = pd.DataFrame(feature_data)
            
            # Create pivot table for heatmap
            pivot_df = df.pivot_table(
                index='feature', 
                columns='clause', 
                values='importance', 
                aggfunc='mean',
                fill_value=0
            )
            
            return {
                'data': pivot_df.to_dict(),
                'features': pivot_df.index.tolist(),
                'clauses': pivot_df.columns.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error creating heatmap data: {e}")
            return {}
    
    def _create_clause_comparison_chart(self, explanations: Dict[str, ExplanationResult]) -> Dict[str, Any]:
        """Create clause comparison chart data"""
        try:
            comparison_data = []
            
            for clause_name, explanation in explanations.items():
                # Calculate summary statistics
                positive_importance = sum(score for _, score in explanation.top_positive_features)
                negative_importance = sum(abs(score) for _, score in explanation.top_negative_features)
                
                comparison_data.append({
                    'clause': clause_name,
                    'confidence': explanation.confidence,
                    'positive_evidence': positive_importance,
                    'negative_evidence': negative_importance,
                    'net_evidence': positive_importance - negative_importance,
                    'method': explanation.method
                })
            
            return {'data': comparison_data}
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            return {}
    
    def _create_attribution_distribution(self, explanations: Dict[str, ExplanationResult]) -> Dict[str, Any]:
        """Create attribution distribution data"""
        try:
            distribution_data = []
            
            for clause_name, explanation in explanations.items():
                attributions = list(explanation.feature_attributions.values())
                
                distribution_data.append({
                    'clause': clause_name,
                    'mean_attribution': np.mean(attributions),
                    'std_attribution': np.std(attributions),
                    'max_attribution': np.max(attributions),
                    'min_attribution': np.min(attributions),
                    'num_features': len(attributions)
                })
            
            return {'data': distribution_data}
            
        except Exception as e:
            logger.error(f"Error creating distribution data: {e}")
            return {}
    
    def _create_wordcloud_data(self, explanations: Dict[str, ExplanationResult]) -> Dict[str, Any]:
        """Create word cloud data for top features"""
        try:
            wordcloud_data = {}
            
            for clause_name, explanation in explanations.items():
                # Combine positive and negative features
                all_features = {}
                
                for feature, importance in explanation.top_positive_features:
                    all_features[feature] = abs(importance)
                
                for feature, importance in explanation.top_negative_features:
                    all_features[feature] = abs(importance) * 0.5  # Reduce weight for negative
                
                wordcloud_data[clause_name] = all_features
            
            return wordcloud_data
            
        except Exception as e:
            logger.error(f"Error creating wordcloud data: {e}")
            return {}
    
    def get_explanation_statistics(self, explanations: Dict[str, ExplanationResult]) -> Dict[str, Any]:
        """Generate statistics about the explanations"""
        if not explanations:
            return {}
        
        try:
            stats = {
                'total_explanations': len(explanations),
                'methods_used': list(set(exp.method for exp in explanations.values())),
                'avg_confidence': np.mean([exp.confidence for exp in explanations.values()]),
                'avg_positive_features': np.mean([len(exp.top_positive_features) for exp in explanations.values()]),
                'avg_negative_features': np.mean([len(exp.top_negative_features) for exp in explanations.values()]),
                'most_important_features': self._get_most_important_global_features(explanations),
                'explanation_coverage': self._calculate_explanation_coverage(explanations)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating explanation statistics: {e}")
            return {}
    
    def _get_most_important_global_features(self, explanations: Dict[str, ExplanationResult]) -> List[Tuple[str, float]]:
        """Get globally most important features across all explanations"""
        feature_scores = {}
        
        for explanation in explanations.values():
            for feature, score in explanation.feature_attributions.items():
                if feature not in feature_scores:
                    feature_scores[feature] = []
                feature_scores[feature].append(abs(score))
        
        # Calculate mean importance for each feature
        global_importance = {
            feature: np.mean(scores) 
            for feature, scores in feature_scores.items()
        }
        
        # Sort and return top features
        return sorted(global_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    
    def _calculate_explanation_coverage(self, explanations: Dict[str, ExplanationResult]) -> Dict[str, float]:
        """Calculate how well explanations cover the predictions"""
        coverage_stats = {}
        
        for clause_name, explanation in explanations.items():
            total_attribution = sum(abs(score) for score in explanation.feature_attributions.values())
            top_features_attribution = sum(abs(score) for _, score in explanation.top_positive_features[:10])
            
            coverage = (top_features_attribution / total_attribution) if total_attribution > 0 else 0
            coverage_stats[clause_name] = coverage
        
        return coverage_stats

# Streamlit Interface Functions
@st.cache_resource
def load_legal_explainer(model=None, tokenizer=None, clause_types=None, clean_clause_names=None):
    """Load and cache the legal explainer"""
    try:
        return LegalExplainer(
            model=model,
            tokenizer=tokenizer,
            clause_types=clause_types,
            clean_clause_names=clean_clause_names
        )
    except Exception as e:
        st.error(f"Error loading explainer: {e}")
        return None

def render_explainability_interface(
    text: str,
    predicted_clauses: List[Dict[str, Any]],
    model=None,
    tokenizer=None,
    clause_types: List[str] = None,
    clean_clause_names: Dict[str, str] = None
):
    """Render the explainability interface in Streamlit"""
    
    st.header("🔍 AI Explainability Analysis")
    st.markdown("""
    **Understand why the AI made its decisions** with comprehensive explainability analysis.
    Choose from multiple explanation methods to gain insights into model predictions.
    """)
    
    if not predicted_clauses:
        st.info("💡 No clause predictions to explain. Please run clause extraction first.")
        return
    
    # Load explainer
    explainer = load_legal_explainer(model, tokenizer, clause_types, clean_clause_names)
    if explainer is None:
        st.error("⚠️ Failed to load explainability framework.")
        return
    
    # Configuration section
    st.subheader("⚙️ Explanation Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        explanation_method = st.selectbox(
            "Explanation Method",
            options=['shap', 'lime', 'integrated_gradients', 'attention'],
            index=0,
            help="Choose the explanation method to use"
        )
    
    with col2:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=0.9, value=0.3, step=0.05,
            help="Only explain clauses above this confidence"
        )
    
    with col3:
        top_k_features = st.slider(
            "Top Features",
            min_value=5, max_value=50, value=20, step=5,
            help="Number of top features to display"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            include_negative = st.checkbox(
                "Include Negative Evidence", 
                value=True,
                help="Show features that argue against the prediction"
            )
            
            normalize_attributions = st.checkbox(
                "Normalize Attributions", 
                value=True,
                help="Normalize feature importance scores"
            )
        
        with col2:
            generate_plots = st.checkbox(
                "Generate Visualizations", 
                value=True,
                help="Create interactive plots and charts"
            )
            
            save_explanations = st.checkbox(
                "Save Explanations", 
                value=False,
                help="Save explanation results for later analysis"
            )
    
    # Create configuration
    config = ExplanationConfig(
        explanation_method=explanation_method,
        top_k_features=top_k_features,
        confidence_threshold=confidence_threshold,
        include_negative_attributions=include_negative,
        normalize_attributions=normalize_attributions,
        generate_plots=generate_plots,
        save_explanations=save_explanations
    )
    
    # Generate explanations
    if st.button("🚀 Generate Explanations", type="primary", use_container_width=True):
        
        # Filter clauses above threshold
        relevant_clauses = [
            clause for clause in predicted_clauses 
            if clause['confidence'] >= confidence_threshold
        ]
        
        if not relevant_clauses:
            st.warning(f"⚠️ No clauses above confidence threshold ({confidence_threshold})")
            return
        
        with st.spinner(f"Generating {explanation_method.upper()} explanations..."):
            explanations = explainer.explain_prediction(text, relevant_clauses, config)
        
        if not explanations:
            st.error("❌ Failed to generate explanations. Please try a different method or check the model.")
            return
        
        st.success(f"✅ Generated explanations for {len(explanations)} clauses!")
        
        # Display explanations
        st.subheader("📋 Explanation Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 Detailed Explanations", 
            "📊 Visual Analysis", 
            "📈 Statistics", 
            "💾 Export Results"
        ])
        
        with tab1:
            # Detailed explanations for each clause
            for clause_name, explanation in explanations.items():
                with st.expander(f"🔍 {clause_name} (Confidence: {explanation.confidence:.3f})"):
                    
                    # Method and summary
                    st.markdown(f"**Method:** {explanation.method}")
                    st.markdown(explanation.explanation_summary)
                    
                    # Feature importance table
                    if explanation.top_positive_features or explanation.top_negative_features:
                        st.markdown("**Feature Importance:**")
                        
                        # Combine positive and negative features
                        feature_data = []
                        
                        for feature, score in explanation.top_positive_features:
                            feature_data.append({
                                'Feature': feature,
                                'Importance': f"{score:.4f}",
                                'Type': '🟢 Supporting',
                                'Abs_Score': abs(score)
                            })
                        
                        if include_negative:
                            for feature, score in explanation.top_negative_features:
                                feature_data.append({
                                    'Feature': feature,
                                    'Importance': f"{score:.4f}",
                                    'Type': '🔴 Contradicting',
                                    'Abs_Score': abs(score)
                                })
                        
                        # Sort by absolute importance
                        feature_data.sort(key=lambda x: x['Abs_Score'], reverse=True)
                        
                        # Display as dataframe
                        feature_df = pd.DataFrame(feature_data).drop('Abs_Score', axis=1)
                        st.dataframe(feature_df, hide_index=True, use_container_width=True)
        
        with tab2:
            # Visual analysis
            st.markdown("### 📊 Visual Explanation Analysis")
            
            if generate_plots:
                visualizations = explainer.create_explanation_visualizations(explanations)
                
                # Feature importance heatmap
                if 'feature_heatmap' in visualizations and visualizations['feature_heatmap']:
                    st.markdown("#### 🔥 Feature Importance Heatmap")
                    
                    heatmap_data = visualizations['feature_heatmap']
                    
                    if heatmap_data.get('data'):
                        # Convert to DataFrame for plotting
                        heatmap_df = pd.DataFrame(heatmap_data['data']).fillna(0)
                        
                        if not heatmap_df.empty:
                            fig = px.imshow(
                                heatmap_df.values,
                                x=heatmap_df.columns,
                                y=heatmap_df.index,
                                color_continuous_scale='RdYlBu_r',
                                title='Feature Importance Across Clauses',
                                aspect='auto'
                            )
                            fig.update_layout(height=max(400, len(heatmap_df) * 25))
                            st.plotly_chart(fig, use_container_width=True)
                
                # Clause comparison chart
                if 'clause_comparison' in visualizations and visualizations['clause_comparison']:
                    st.markdown("#### ⚖️ Clause Evidence Comparison")
                    
                    comparison_data = visualizations['clause_comparison']['data']
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    if not comparison_df.empty:
                        fig = px.bar(
                            comparison_df,
                            x='clause',
                            y=['positive_evidence', 'negative_evidence'],
                            title='Supporting vs Contradicting Evidence',
                            barmode='group',
                            color_discrete_map={
                                'positive_evidence': '#2E8B57',
                                'negative_evidence': '#CD5C5C'
                            }
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Attribution distribution
                if 'attribution_distribution' in visualizations:
                    st.markdown("#### 📈 Attribution Distribution")
                    
                    dist_data = visualizations['attribution_distribution']['data']
                    dist_df = pd.DataFrame(dist_data)
                    
                    if not dist_df.empty:
                        fig = px.box(
                            dist_df,
                            y='clause',
                            x='mean_attribution',
                            title='Attribution Distribution by Clause',
                            orientation='h'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("📊 Enable 'Generate Visualizations' to see interactive charts.")
        
        with tab3:
            # Statistics
            st.markdown("### 📈 Explanation Statistics")
            
            stats = explainer.get_explanation_statistics(explanations)
            
            if stats:
                # Overview metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Explanations", stats['total_explanations'])
                with col2:
                    st.metric("Methods Used", len(stats['methods_used']))
                with col3:
                    st.metric("Avg Confidence", f"{stats['avg_confidence']:.3f}")
                with col4:
                    st.metric("Avg Features", f"{stats['avg_positive_features']:.1f}")
                
                # Most important global features
                if stats.get('most_important_features'):
                    st.markdown("#### 🌟 Most Important Global Features")
                    
                    global_features = stats['most_important_features'][:15]
                    global_df = pd.DataFrame(global_features, columns=['Feature', 'Global Importance'])
                    global_df['Global Importance'] = global_df['Global Importance'].round(4)
                    
                    st.dataframe(global_df, hide_index=True, use_container_width=True)
                
                # Explanation coverage
                if stats.get('explanation_coverage'):
                    st.markdown("#### 📊 Explanation Coverage")
                    
                    coverage_data = [
                        {'Clause': clause, 'Coverage': f"{coverage:.1%}"}
                        for clause, coverage in stats['explanation_coverage'].items()
                    ]
                    coverage_df = pd.DataFrame(coverage_data)
                    st.dataframe(coverage_df, hide_index=True, use_container_width=True)
        
        with tab4:
            # Export results
            st.markdown("### 💾 Export Explanation Results")
            
            # Prepare export data
            export_data = {
                'explanation_metadata': {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'method': explanation_method,
                    'configuration': asdict(config),
                    'statistics': stats
                },
                'explanations': {
                    clause_name: explanation.to_dict() 
                    for clause_name, explanation in explanations.items()
                },
                'input_text': text,
                'predicted_clauses': relevant_clauses
            }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📄 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"explanations_{explanation_method}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # Create summary CSV
                if explanations:
                    summary_data = []
                    for clause_name, explanation in explanations.items():
                        for feature, importance in explanation.top_positive_features[:10]:
                            summary_data.append({
                                'Clause': clause_name,
                                'Feature': feature,
                                'Importance': importance,
                                'Type': 'Positive',
                                'Method': explanation.method
                            })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        csv_data = summary_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📊 Download CSV",
                            data=csv_data,
                            file_name=f"explanation_features_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            
            with col3:
                # Generate comprehensive report
                report_lines = [
                    f"# Legal AI Explainability Report",
                    f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Method: {explanation_method.upper()}",
                    f"",
                    f"## Summary",
                    f"- Clauses explained: {len(explanations)}",
                    f"- Confidence threshold: {confidence_threshold}",
                    f"- Average confidence: {stats.get('avg_confidence', 0):.3f}",
                    f"",
                    f"## Key Findings"
                ]
                
                for clause_name, explanation in explanations.items():
                    report_lines.append(f"### {clause_name}")
                    report_lines.append(f"- Confidence: {explanation.confidence:.3f}")
                    report_lines.append(f"- Method: {explanation.method}")
                    
                    if explanation.top_positive_features:
                        top_feature = explanation.top_positive_features[0]
                        report_lines.append(f"- Key evidence: '{top_feature[0]}' ({top_feature[1]:.3f})")
                    
                    report_lines.append("")
                
                report_text = "\n".join(report_lines)
                
                st.download_button(
                    label="📝 Download Report",
                    data=report_text,
                    file_name=f"explainability_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

# Legacy compatibility functions
def explain_predictions(text: str, model, tokenizer, predicted_clauses: List[Dict]) -> Dict[str, Any]:
    """Legacy function for backward compatibility"""
    explainer = LegalExplainer(model=model, tokenizer=tokenizer)
    return explainer.explain_prediction(text, predicted_clauses)

# Main execution for testing
if __name__ == "__main__":
    print("Testing Enhanced Legal Explainer...")
    
    # This would normally be run with actual model and tokenizer
    sample_text = """
    This Software License Agreement is effective as of January 1, 2024 
    between TechCorp Inc. and ClientCorp LLC. The agreement will expire 
    on December 31, 2026. This Agreement shall be governed by the laws 
    of the State of California.
    """
    
    sample_clauses = [
        {'clause_type': 'agreement_date', 'clean_name': 'Agreement Date', 'confidence': 0.85},
        {'clause_type': 'expiration_date', 'clean_name': 'Expiration Date', 'confidence': 0.72},
        {'clause_type': 'governing_law', 'clean_name': 'Governing Law', 'confidence': 0.68}
    ]
    
    # Initialize explainer (would need actual model/tokenizer)
    explainer = LegalExplainer()
    
    print(f"Explainer initialized with methods: {['SHAP', 'LIME', 'Integrated Gradients', 'Attention']}")
    print(f"Sample clauses for explanation: {len(sample_clauses)}")
    print("For full functionality, integrate with trained BERT model and tokenizer.")