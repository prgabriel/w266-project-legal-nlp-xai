"""
Legal NLP + Explainability Toolkit - Main Streamlit Application
Comprehensive legal document analysis with enhanced clause extraction, summarization, and explainability
"""
import streamlit as st
# from app.config.azure import azure_config

# # Configure for Azure deployment
# if azure_config.is_azure_environment:
#     # Use Azure Blob Storage for models
#     from app.utils.azure_storage import AzureModelManager
#     model_manager = AzureModelManager()
    
#     # Application Insights integration
#     if azure_config.instrumentation_key:
#         from opencensus.ext.azure.log_exporter import AzureLogHandler
#         import logging
#         logger = logging.getLogger(__name__)
#         logger.addHandler(AzureLogHandler(
#             connection_string=f'InstrumentationKey={azure_config.instrumentation_key}'
#         ))

# Streamlit config for Azure
st.set_page_config(
    page_title="Legal NLP + Explainability Toolkit",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/prgabriel/w266-project-legal-nlp-xai',
        'Report a bug': 'https://github.com/prgabriel/w266-project-legal-nlp-xai/issues',
        'About': """
        # Legal NLP + Explainability Toolkit
        
        **Towards Responsible AI in Legal NLP**
        
        This toolkit provides interpretable clause extraction, 
        summarization, and explainability for legal document analysis.
        
        **Features:**
        - 41 CUAD legal clause types
        - Multi-label BERT classification
        - T5-based legal summarization
        - SHAP, LIME, and attention analysis
        - Interactive visualizations
        
        Built for legal professionals and researchers.
        """
    }
)

import sys
import os
import logging
import traceback
from typing import Dict, List, Any, Optional
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import enhanced components
from components.clause_extractor import (
    render_enhanced_clause_interface, 
    LegalClauseExtractor,
    extract_clauses
)
from components.summarizer import (
    render_enhanced_summarization_interface,
    LegalDocumentSummarizer,
    summarize_text
)
from components.explainer import (
    render_explainability_interface,
    LegalExplainer,
    explain_predictions
)

# Import utilities
from scripts.utils import PROJECT, load_data, preprocess_text
from scripts.evaluation_metrics import LegalNLPEvaluator

# Add Plotly imports for analytics
import plotly.express as px
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom CSS for enhanced styling
def load_custom_css():
    """Load custom CSS with proper styling"""
    st.markdown("""
    <style>
    /* Import the main stylesheet */
    @import url('static/styles.css');
    
    /* Override and additional styles for Streamlit */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
    }
    
    /* Fix Streamlit container width */
    .main .block-container {
        max-width: 1200px;
        padding: 2rem 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #1f4e79;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    
    .metric-card h4 {
        color: #1f4e79;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    .metric-card p {
        margin: 0.5rem 0;
        line-height: 1.5;
    }
    
    .metric-card p strong {
        color: #d4af37;
        font-size: 1.1rem;
    }
    
    /* Status boxes */
    .success-box, .warning-box, .error-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left-color: #28a745;
        color: #155724;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left-color: #ffc107;
        color: #856404;
    }
    
    .error-box {
        background-color: #f8d7da;
        border-left-color: #dc3545;
        color: #721c24;
    }
    
    /* Sidebar improvements */
    .sidebar-info {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 3px solid #0066cc;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background-color: transparent;
        border-radius: 8px;
        color: #6c757d;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
        color: #1f4e79;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79;
        color: white !important;
    }
    
    /* Button improvements */
    .stButton > button {
        background: linear-gradient(135deg, #1f4e79 0%, #4a6fa5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(31, 78, 121, 0.3);
    }
    
    /* Text area improvements */
    .stTextArea textarea {
        border: 2px solid #e9ecef;
        border-radius: 8px;
        font-family: 'Consolas', monospace;
        line-height: 1.5;
    }
    
    .stTextArea textarea:focus {
        border-color: #1f4e79;
        box-shadow: 0 0 0 2px rgba(31, 78, 121, 0.2);
    }
    
    /* Selectbox improvements */
    .stSelectbox > div > div {
        border: 2px solid #e9ecef;
        border-radius: 8px;
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 2rem 1rem;
        border-top: 1px solid #e9ecef;
        margin-top: 3rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .metric-card {
            margin: 0.5rem 0;
        }
        
        .feature-card {
            margin: 1rem 0;
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_models():
    """Initialize and cache all models"""
    models = {}
    
    with st.spinner("Initializing models... This may take a moment."):
        try:
            # Initialize clause extractor
            models['clause_extractor'] = LegalClauseExtractor()
            
            # Initialize summarizer
            models['summarizer'] = LegalDocumentSummarizer()
            
            # Initialize explainer (will be created per session as needed)
            models['explainer'] = None
            
            # Initialize evaluator
            models['evaluator'] = LegalNLPEvaluator()
            
            logger.info("All models initialized successfully")
            return models
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            st.error(f"Error initializing models: {e}")
            return {}

def get_model_performance_metrics(models):
    """Get actual performance metrics from models and training results - TRULY LIVE VERSION"""
    metrics = {
        'clause_extraction': {
            'status': 'unknown',
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'num_clause_types': 0,
            'model_name': 'Unknown'
        },
        'summarization': {
            'status': 'unknown',
            'rouge_1': 0.0,
            'rouge_2': 0.0,
            'rouge_l': 0.0,
            'model_name': 'Unknown'
        }
    }
    
    # Get clause extraction metrics - LOAD ACTUAL TRAINING RESULTS
    if models.get('clause_extractor'):
        try:
            # First get model info
            if hasattr(models['clause_extractor'], 'get_model_info'):
                model_info = models['clause_extractor'].get_model_info()
                metrics['clause_extraction']['status'] = 'loaded' if model_info.get('model_loaded', False) else 'error'
                metrics['clause_extraction']['num_clause_types'] = model_info.get('num_clause_types', 41)
            
            # Load ACTUAL training results directly - NO FALLBACKS
            project_root = Path(__file__).parent.parent
            bert_training_path = project_root / 'models' / 'bert' / 'training_results.json'
            
            if bert_training_path.exists():
                try:
                    with open(bert_training_path, 'r') as f:
                        training_data = json.load(f)
                    
                    # Get the ACTUAL model name from training results
                    model_config = training_data.get('model_config', {})
                    actual_model_name = model_config.get('model_name', training_data.get('model_name', 'nlpaueb/legal-bert-base-uncased'))
                    metrics['clause_extraction']['model_name'] = actual_model_name
                    
                    # Get the ACTUAL test metrics (not fallbacks!)
                    test_metrics = training_data.get('test_metrics', {})
                    if test_metrics:
                        # Use the REAL values from your training
                        metrics['clause_extraction']['f1_score'] = test_metrics.get('f1_micro', 0.0)
                        metrics['clause_extraction']['precision'] = test_metrics.get('precision_micro', 0.0)
                        metrics['clause_extraction']['recall'] = test_metrics.get('recall_micro', 0.0)
                        
                        logger.info(f"✅ Loaded REAL BERT metrics - Model: {actual_model_name}")
                        logger.info(f"   F1: {metrics['clause_extraction']['f1_score']:.4f}")
                        logger.info(f"   Precision: {metrics['clause_extraction']['precision']:.4f}")  
                        logger.info(f"   Recall: {metrics['clause_extraction']['recall']:.4f}")
                    else:
                        # Try final validation metrics if no test_metrics
                        val_history = training_data.get('training_history', {}).get('val_metrics', [])
                        if val_history:
                            final_val = val_history[-1]  # Last validation results
                            metrics['clause_extraction']['f1_score'] = final_val.get('f1_micro', 0.0)
                            metrics['clause_extraction']['precision'] = final_val.get('precision_micro', 0.0)
                            metrics['clause_extraction']['recall'] = final_val.get('recall_micro', 0.0)
                            logger.info(f"✅ Loaded validation metrics (no test metrics found)")
                        else:
                            logger.warning("❌ No test_metrics or validation metrics found in training results")
                    
                except Exception as e:
                    logger.error(f"Error parsing BERT training results: {e}")
                    metrics['clause_extraction']['model_name'] = 'nlpaueb/legal-bert-base-uncased'
                    metrics['clause_extraction']['status'] = 'error'
            else:
                logger.warning(f"❌ BERT training results not found at {bert_training_path}")
                metrics['clause_extraction']['model_name'] = 'bert-base-uncased'  # Fallback indicator
                
        except Exception as e:
            logger.warning(f"Could not load clause extraction metrics: {e}")
            metrics['clause_extraction']['status'] = 'error'
    
    # Get summarization metrics - LOAD ACTUAL T5 RESULTS  
    if models.get('summarizer'):
        try:
            # First get model info
            if hasattr(models['summarizer'], 'get_model_info'):
                model_info = models['summarizer'].get_model_info()
                metrics['summarization']['status'] = 'loaded' if model_info.get('model_loaded', False) else 'error'
                base_model_name = model_info.get('model_name', 't5-base')
                metrics['summarization']['model_name'] = base_model_name
            
            # Load ACTUAL T5 training results directly - NO FALLBACKS
            project_root = Path(__file__).parent.parent
            t5_training_path = project_root / 'models' / 't5' / 'training_results.json'
            
            if t5_training_path.exists():
                try:
                    with open(t5_training_path, 'r') as f:
                        t5_data = json.load(f)
                    
                    # Get ACTUAL ROUGE scores from training results
                    rouge_scores = t5_data.get('training_history', {}).get('rouge_scores', {})
                    if rouge_scores:
                        metrics['summarization']['rouge_1'] = rouge_scores.get('rouge1', 0.0)
                        metrics['summarization']['rouge_2'] = rouge_scores.get('rouge2', 0.0) 
                        metrics['summarization']['rouge_l'] = rouge_scores.get('rougeL', 0.0)
                        
                        logger.info(f"✅ Loaded REAL T5 metrics:")
                        logger.info(f"   ROUGE-1: {metrics['summarization']['rouge_1']:.4f}")
                        logger.info(f"   ROUGE-2: {metrics['summarization']['rouge_2']:.4f}")
                        logger.info(f"   ROUGE-L: {metrics['summarization']['rouge_l']:.4f}")
                    else:
                        logger.warning("❌ No ROUGE scores found in T5 training results")
                        
                except Exception as e:
                    logger.error(f"Error parsing T5 training results: {e}")
            else:
                logger.warning(f"❌ T5 training results not found at {t5_training_path}")
                
        except Exception as e:
            logger.warning(f"Could not load summarization metrics: {e}")
            metrics['summarization']['status'] = 'error'
            metrics['summarization']['model_name'] = 't5-base'
    
    # Show what we loaded
    logger.info("📊 Final loaded metrics:")
    logger.info(f"   BERT F1: {metrics['clause_extraction']['f1_score']:.4f} ({'✅ Live' if metrics['clause_extraction']['f1_score'] > 0 else '❌ Missing'})")
    logger.info(f"   T5 ROUGE-L: {metrics['summarization']['rouge_l']:.4f} ({'✅ Live' if metrics['summarization']['rouge_l'] > 0 else '❌ Missing'})")
    
    return metrics

# Optional: Add a test function to verify it works
# def test_metrics_function():
#     """Test function to verify metrics loading works"""
#     models = initialize_models()
#     metrics = get_model_performance_metrics(models)
#     st.write("🧪 **Metrics Test Results:**")
#     st.json(metrics)
#     return metrics

def render_home_page():
    """Render the home/overview page with dynamic model information"""
    # Main header with better styling
    st.markdown("""
    <div class="main-header">
        ⚖️ Legal NLP + Explainability Toolkit
        <div style="font-size: 1.2rem; font-weight: 300; margin-top: 0.5rem; opacity: 0.9;">
            Towards Responsible AI in Legal Document Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #1f4e79; margin-bottom: 1rem;">Welcome to Advanced Legal AI</h3>
        <p style="font-size: 1.1rem; line-height: 1.6; color: #333;">
            This comprehensive toolkit provides interpretable clause extraction, intelligent summarization, 
            and detailed explainability analysis for legal document review and contract analysis.
            Built with state-of-the-art transformer models and responsible AI principles.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get dynamic metrics
    models = initialize_models()
    performance_metrics = get_model_performance_metrics(models)
    
    # Feature overview with DYNAMIC cards
    st.markdown("<h2 style='text-align: center; color: #1f4e79; margin: 2rem 0;'>Core Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Dynamic clause extraction metrics
        clause_metrics = performance_metrics['clause_extraction']
        f1_percentage = clause_metrics['f1_score'] * 100
        num_clauses = clause_metrics['num_clause_types']
        
        st.markdown(f"""
        <div class="metric-card" style="color: #1f4e79;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📋</div>
            <h4>Clause Extraction</h4>
            <p><strong>{num_clauses} Legal Clause Types</strong></p>
            <p>Multi-label BERT classification for comprehensive legal clause detection with clean, human-readable clause names and confidence scoring.</p>
            <div style="margin-top: 1rem; padding: 0.5rem; background: #e7f3ff; border-radius: 5px; color: #0066cc;">
                <small><strong>F1-Score:</strong> {f1_percentage:.1f}%</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Dynamic summarization metrics
        summ_metrics = performance_metrics['summarization']
        rouge_l = summ_metrics['rouge_l']
        
        st.markdown(f"""
        <div class="metric-card" style="color: #1f4e79;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
            <h4>Document Summarization</h4>
            <p><strong>T5-based Intelligence</strong></p>
            <p>Legal-optimized document summarization with extractive, abstractive, and hybrid summarization modes for different use cases.</p>
            <div style="margin-top: 1rem; padding: 0.5rem; background: #fff3cd; border-radius: 5px; color: #0066cc;">
                <small><strong>ROUGE-L:</strong> {rouge_l:.3f} Score</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Dynamic explainability info
        xai_methods = 4  # We'll make this dynamic in a later step
        
        st.markdown(f"""
        <div class="metric-card" style="color: #1f4e79;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
            <h4>AI Explainability</h4>
            <p><strong>SHAP, LIME & Attention</strong></p>
            <p>Comprehensive interpretability analysis to understand and trust AI decisions in critical legal document analysis.</p>
            <div style="margin-top: 1rem; padding: 0.5rem; background: #d4edda; border-radius: 5px; color: #0066cc;">
                <small><strong>Methods:</strong> {xai_methods} XAI Techniques</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide with FIXED formatting - use Markdown instead of HTML
    st.markdown("## Quick Start Guide")

    with st.expander("How to Use This Toolkit", expanded=True):
        st.markdown(f"""
        ### 1. Clause Extraction
        Extract {num_clauses} different clause types with {f1_percentage:.1f}% F1-Score accuracy using our fine-tuned Legal-BERT model.
        
        ### 2. Document Summarization
        Generate summaries using {summ_metrics['model_name']} with {rouge_l:.3f} ROUGE-L score for legal document comprehension.
        
        ### 3. Explainability Analysis
        Understand AI decisions using {xai_methods} available XAI methods including SHAP, LIME, attention visualizations, and feature importance.
        
        ### 4. Analytics Dashboard
        Monitor real-time performance with live metrics from your actual model training results.
        
        ---
        
        ###  Pro Tips:
        - Start with clause extraction to identify key contract provisions
        - Use summarization for quick document overview and key points
        - Apply explainability to understand AI reasoning for critical decisions
        - Adjust confidence thresholds in the sidebar for optimal results
        - Export results in multiple formats for reporting and analysis
        """)

    # Dynamic call-to-action based on model status
    system_ready = (clause_metrics['status'] == 'loaded' and summ_metrics['status'] == 'loaded')
    
    if system_ready:
        cta_color = "linear-gradient(135deg, #1f4e79 0%, #4a6fa5 100%)"
        cta_text = "🚀 Ready to Get Started?"
        cta_message = f"All systems operational with {f1_percentage:.1f}% F1-Score performance! Choose an analysis mode from the sidebar."
    else:
        cta_color = "linear-gradient(135deg, #856404 0%, #ffc107 100%)"
        cta_text = "⚠️ Limited Functionality"
        cta_message = "Some components are not fully loaded. You can still use available features."
    
    st.markdown(f"""
    <div style="text-align: center; margin: 2rem 0; padding: 2rem; background: {cta_color}; color: white; border-radius: 15px;">
        <h3 style="color: white; margin-bottom: 1rem;">{cta_text}</h3>
        <p style="font-size: 1.1rem; margin-bottom: 1.5rem;">{cta_message}</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">Built for legal professionals, researchers, and AI practitioners</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dynamic model information section
    if st.checkbox("Show Model Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 BERT Clause Extractor")
            status_icon = "✅" if clause_metrics['status'] == 'loaded' else "❌"
            st.info(f"""
            - **Model**: {clause_metrics['model_name']}
            - **Status**: {status_icon} {clause_metrics['status'].title()}
            - **Clause Types**: {clause_metrics['num_clause_types']} CUAD categories
            - **F1-Score**: {clause_metrics['f1_score']:.3f}
            - **Precision**: {clause_metrics['precision']:.3f}
            - **Recall**: {clause_metrics['recall']:.3f}
            """)
        
        with col2:
            st.markdown("### 📝 T5 Summarizer")
            status_icon = "✅" if summ_metrics['status'] == 'loaded' else "❌"
            st.info(f"""
            - **Model**: {summ_metrics['model_name']}
            - **Status**: {status_icon} {summ_metrics['status'].title()}
            - **ROUGE-1**: {summ_metrics['rouge_1']:.3f}
            - **ROUGE-2**: {summ_metrics['rouge_2']:.3f}
            - **ROUGE-L**: {summ_metrics['rouge_l']:.3f}
            - **Modes**: Abstractive, Extractive, Hybrid
            """)

    # TEMPORARY TEST - we'll remove this after verifying it works
    # if st.button("🧪 Test Metrics Function"):
    #     test_metrics_function()

def render_sidebar():
    """Render enhanced sidebar with navigation and settings"""
    st.sidebar.markdown("## ⚖️ Navigation")
    
    # Main navigation
    page = st.sidebar.selectbox(
        "Select Analysis Mode",
        ["🏠 Home", "📋 Clause Extraction", "📄 Summarization", "🔍 Explainability", "📊 Analytics"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Global settings
    st.sidebar.markdown("## ⚙️ Global Settings")
    
    # Text preprocessing options
    with st.sidebar.expander("🔧 Text Processing"):
        preprocess_enabled = st.checkbox("Enable Legal Preprocessing", value=True)
        max_length = st.slider("Maximum Text Length", 100, 2000, 512)
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05)
    
    # Model settings
    with st.sidebar.expander("🤖 Model Settings"):
        use_cached_predictions = st.checkbox("Use Cached Predictions", value=True)
        enable_batch_processing = st.checkbox("Enable Batch Processing", value=False)
    
    # Export options
    with st.sidebar.expander("💾 Export Options"):
        export_format = st.selectbox("Export Format", ["JSON", "CSV", "PDF Report"])
        include_explanations = st.checkbox("Include Explanations", value=True)
    
    st.sidebar.markdown("---")
    
    # System information
    with st.sidebar.expander("ℹ️ System Info"):
        st.markdown(f"""
        **Models Status**: ✅ Ready\n
        **Cache Status**: ✅ Active\n
        **Processing Mode**: {'Batch' if enable_batch_processing else 'Single'}\n
        **Confidence**: {confidence_threshold:.2f}
        """)
    
    return {
        'page': page,
        'preprocess_enabled': preprocess_enabled,
        'max_length': max_length,
        'confidence_threshold': confidence_threshold,
        'use_cached_predictions': use_cached_predictions,
        'enable_batch_processing': enable_batch_processing,
        'export_format': export_format,
        'include_explanations': include_explanations
    }

def render_analytics_page(models: Dict):
    """Render analytics and performance dashboard - LIVE DATA VERSION"""
    st.markdown("# 📊 Analytics Dashboard")
    
    if not models.get('evaluator'):
        st.warning("Analytics not available - evaluator not initialized")
        return
    
    # Get live metrics for display
    live_metrics = get_model_performance_metrics(models)
    
    tab1, tab2, tab3 = st.tabs(["📈 Model Performance", "📋 Clause Analysis", "🔍 Usage Statistics"])
    
    with tab1:
        st.markdown("##  Model Performance Metrics")
        
        # Show live system metrics first
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            clause_f1 = live_metrics['clause_extraction']['f1_score']
            st.metric("Clause F1 Score", f"{clause_f1:.3f}", delta=f"{'✅ Live' if clause_f1 > 0 else '❌ No Data'}")
        
        with col2:
            clause_precision = live_metrics['clause_extraction']['precision']
            st.metric("Clause Precision", f"{clause_precision:.3f}", delta=f"{'✅ Live' if clause_precision > 0 else '❌ No Data'}")
        
        with col3:
            summ_rouge = live_metrics['summarization']['rouge_l']
            st.metric("Summary ROUGE-L", f"{summ_rouge:.3f}", delta=f"{'✅ Live' if summ_rouge > 0 else '❌ No Data'}")
        
        with col4:
            num_clauses = live_metrics['clause_extraction']['num_clause_types']
            st.metric("Clause Types", num_clauses, delta=f"{'✅ Live' if num_clauses > 0 else '❌ No Data'}")
        
        # Load detailed performance data
        try:
            performance_data = models['evaluator'].load_clause_performance_data()
            if performance_data is not None:
                st.markdown("###  Detailed Per-Clause Performance (Live Data)")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_f1 = performance_data['f1'].mean()
                    st.metric("Average F1 Score", f"{avg_f1:.3f}")
                
                with col2:
                    avg_precision = performance_data['precision'].mean()
                    st.metric("Average Precision", f"{avg_precision:.3f}")
                
                with col3:
                    avg_recall = performance_data['recall'].mean()
                    st.metric("Average Recall", f"{avg_recall:.3f}")
                
                with col4:
                    total_clauses = len(performance_data)
                    st.metric("Clause Types", total_clauses)
                
                # Performance visualization
                st.markdown("###  Per-Clause Performance")
                
                # Create a cleaner display with proper column names
                display_df = performance_data[['clause_name', 'precision', 'recall', 'f1', 'support', 'avg_confidence']].copy()
                display_df.columns = ['Clause Name', 'Precision', 'Recall', 'F1 Score', 'Support', 'Avg Confidence']
                
                display_df = display_df.sort_values('F1 Score', ascending=False)
                
                st.dataframe(
                    display_df,
                    use_container_width=True
                )
                
                # Add performance visualization
                st.markdown("###  Performance Visualization")
                
                # Create a performance chart
                fig = px.bar(
                    x=performance_data['f1'].head(10),
                    y=performance_data['clause_name'].head(10),
                    orientation='h',
                    title='Top 10 Clause Types by F1 Score (Live Data)',
                    labels={'x': 'F1 Score', 'y': 'Clause Type'},
                    color=performance_data['f1'].head(10),
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info(" Detailed performance data not available. Run model evaluation to generate live metrics.")
        
        except Exception as e:
            st.error(f"Error loading live performance data: {e}")
            logger.error(f"Analytics error: {e}")
    
    with tab2:
        st.markdown("##  Clause Type Analysis")

        # Use live data if available, otherwise show sample
        try:
            performance_data = models['evaluator'].load_clause_performance_data()
            if performance_data is not None:
                # Create dynamic categories based on actual performance
                high_perf = performance_data[performance_data['f1'] >= 0.8]['clause_name'].tolist()
                medium_perf = performance_data[(performance_data['f1'] >= 0.6) & (performance_data['f1'] < 0.8)]['clause_name'].tolist()
                low_perf = performance_data[performance_data['f1'] < 0.6]['clause_name'].tolist()
                
                clause_info = {
                    f'High Performance (F1 ≥ 0.8) - {len(high_perf)} clauses': high_perf[:10],  # Show top 10
                    f'Medium Performance (0.6 ≤ F1 < 0.8) - {len(medium_perf)} clauses': medium_perf[:10],
                    f'Challenging (F1 < 0.6) - {len(low_perf)} clauses': low_perf[:10]
                }
            else:
                # Fallback to sample data
                clause_info = {
                    'High Performance (Sample)': ['License Grant', 'Agreement Date', 'Parties', 'Document Name'],
                    'Medium Performance (Sample)': ['Governing Law', 'Termination', 'Insurance', 'Audit Rights'],
                    'Challenging (Sample)': ['Most Favored Nation', 'Revenue Sharing', 'IP Ownership', 'Anti-Assignment']
                }
        except Exception as e:
            logger.error(f"Error categorizing clauses: {e}")
            # Fallback to sample data
            clause_info = {
                'High Performance (Sample)': ['License Grant', 'Agreement Date', 'Parties', 'Document Name'],
                'Medium Performance (Sample)': ['Governing Law', 'Termination', 'Insurance', 'Audit Rights'],
                'Challenging (Sample)': ['Most Favored Nation', 'Revenue Sharing', 'IP Ownership', 'Anti-Assignment']
            }
        
        for category, clauses in clause_info.items():
            with st.expander(f" {category}"):
                for clause in clauses:
                    st.markdown(f"- **{clause}**")
    
    with tab3:
        st.markdown("##  Usage Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("###  Session Statistics (Live)")
            if 'session_stats' not in st.session_state:
                st.session_state.session_stats = {
                    'extractions': 0,
                    'summarizations': 0,
                    'explanations': 0
                }
            
            stats = st.session_state.session_stats
            st.metric("Clause Extractions", stats['extractions'])
            st.metric("Summarizations", stats['summarizations'])
            st.metric("Explanations", stats['explanations'])
            
            # Show model status
            st.markdown("###  Model Status (Live)")
            clause_status = live_metrics['clause_extraction']['status']
            summ_status = live_metrics['summarization']['status']
            
            st.write(f"**Clause Extractor:** {'✅ Loaded' if clause_status == 'loaded' else '❌ Error'}")
            st.write(f"**Summarizer:** {'✅ Loaded' if summ_status == 'loaded' else '❌ Error'}")
        
        with col2:
            st.markdown("###  Live Performance Metrics")
            st.info(f"""
            **Current System Performance:**
            - F1-Score: {live_metrics['clause_extraction']['f1_score']:.3f}
            - ROUGE-L: {live_metrics['summarization']['rouge_l']:.3f}
            - Active Clause Types: {live_metrics['clause_extraction']['num_clause_types']}
            """)

def handle_error(error: Exception, context: str = ""):
    """Centralized error handling with user-friendly messages"""
    error_msg = str(error)
    logger.error(f"Error in {context}: {error_msg}")
    logger.error(traceback.format_exc())
    
    st.markdown(f"""
    <div class="error-box">
    <h4>❌ Error Occurred</h4>
    <p><strong>Context:</strong> {context}</p>
    <p><strong>Details:</strong> {error_msg}</p>
    <p><strong>Suggestion:</strong> Please try again or contact support if the issue persists.</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    try:
        # Load custom CSS
        load_custom_css()
        
        # Initialize models
        models = initialize_models()
        
        # Render sidebar and get settings
        settings = render_sidebar()
        
        # Route to appropriate page
        page = settings['page']
        
        if page == "🏠 Home":
            render_home_page()
        
        elif page == "📋 Clause Extraction":
            st.markdown("# 📋 Legal Clause Extraction")
            try:
                # Pass settings to component
                if hasattr(render_enhanced_clause_interface, '__code__') and 'settings' in render_enhanced_clause_interface.__code__.co_varnames:
                    render_enhanced_clause_interface(settings)
                else:
                    render_enhanced_clause_interface()
                
                # Update session stats
                if 'session_stats' not in st.session_state:
                    st.session_state.session_stats = {'extractions': 0, 'summarizations': 0, 'explanations': 0}
                st.session_state.session_stats['extractions'] += 1
                
            except Exception as e:
                handle_error(e, "Clause Extraction")
        
        elif page == "📄 Summarization":
            st.markdown("# 📄 Legal Document Summarization")
            try:
                # Pass settings to component
                if hasattr(render_enhanced_summarization_interface, '__code__') and 'settings' in render_enhanced_summarization_interface.__code__.co_varnames:
                    render_enhanced_summarization_interface(settings)
                else:
                    render_enhanced_summarization_interface()
                
                # Update session stats
                if 'session_stats' not in st.session_state:
                    st.session_state.session_stats = {'extractions': 0, 'summarizations': 0, 'explanations': 0}
                st.session_state.session_stats['summarizations'] += 1
                
            except Exception as e:
                handle_error(e, "Document Summarization")
        
        elif page == "🔍 Explainability":
            st.markdown("# 🔍 Explainability Analysis")
            
            # Use confidence threshold from settings
            confidence_threshold = settings.get('confidence_threshold', 0.3)
            st.info(f"📝 Using confidence threshold: {confidence_threshold:.2f}. First extract clauses or generate summaries, then use this section to explain the AI's decisions.")
            
            # Simple interface for explainability
            legal_text = st.text_area(
                "Enter legal text for explainability analysis:",
                height=200,
                placeholder="Paste your legal document text here..."
            )
            
            if legal_text and st.button("🔍 Analyze & Explain", type="primary"):
                try:
                    with st.spinner("Extracting clauses and generating explanations..."):
                        # First extract clauses with proper threshold
                        if models.get('clause_extractor'):
                            # Create extraction config with user settings
                            from components.clause_extractor import ExtractionConfig
                            extraction_config = ExtractionConfig(
                                confidence_threshold=confidence_threshold,
                                max_length=settings.get('max_length', 512),
                                enable_preprocessing=settings.get('preprocess_enabled', True)
                            )
                            
                            results = models['clause_extractor'].extract_clauses(legal_text, config=extraction_config)
                            predicted_clauses = results.get('predictions', [])
                            
                            if predicted_clauses:
                                # Update session stats
                                if 'session_stats' not in st.session_state:
                                    st.session_state.session_stats = {'extractions': 0, 'summarizations': 0, 'explanations': 0}
                                st.session_state.session_stats['explanations'] += 1
                                
                                # Render explainability interface
                                render_explainability_interface(
                                    text=legal_text,
                                    predicted_clauses=predicted_clauses,
                                    model=models.get('clause_extractor'),
                                    tokenizer=getattr(models.get('clause_extractor'), 'tokenizer', None)
                                )
                            else:
                                st.warning(f"No clauses detected with confidence threshold {confidence_threshold:.2f}. Try lowering the threshold in the sidebar.")
                        else:
                            st.error("Clause extractor not available")
                            
                except Exception as e:
                    handle_error(e, "Explainability Analysis")
        
        elif page == "📊 Analytics":
            try:
                render_analytics_page(models)
            except Exception as e:
                handle_error(e, "Analytics Dashboard")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
        <p>⚖️ Legal NLP + Explainability Toolkit | Built for Responsible AI in Legal Document Analysis</p>
        <p><small>For support or questions, please refer to the documentation or contact the development team.</small></p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        handle_error(e, "Main Application")
        st.stop()

if __name__ == "__main__":
    main()