"""
Plotting utilities for Legal NLP Explainability Project

This module provides reusable functions for consistent styling and 
visualization across all notebooks and scripts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Optional


# Color schemes and styling constants
COLORS = {
    'input': '#E8F4FD',      # Light blue
    'processing': '#B8E6B8',  # Light green  
    'model': '#FFD93D',       # Yellow
    'output': '#FFB3BA',      # Light pink
    'explainability': '#DDA0DD', # Plum
    'deployment': '#F0E68C',  # Khaki
    'accent': '#FF6B6B',      # Red accent
    'text': '#2C3E50'         # Dark blue-gray
}

STYLE_CONFIG = {
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': True,
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.family': 'serif'
}


def setup_plotting_style(style: str = 'academic') -> None:
    """
    Set up consistent plotting style across all visualizations.
    
    Args:
        style: Style preset ('academic', 'presentation', 'minimal')
    """
    if style == 'academic':
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams.update(STYLE_CONFIG)
    elif style == 'presentation':
        plt.style.use('seaborn-v0_8-dark')
        plt.rcParams.update(STYLE_CONFIG)
        plt.rcParams['font.size'] = 14
    else:  # minimal
        plt.style.use('default')
        plt.rcParams.update({'figure.figsize': (10, 6)})


def create_color_palette(n_colors: int, palette_name: str = 'husl') -> List[str]:
    """
    Create a consistent color palette for visualizations.
    
    Args:
        n_colors: Number of colors needed
        palette_name: Name of the seaborn palette
        
    Returns:
        List of color hex codes
    """
    return sns.color_palette(palette_name, n_colors).as_hex()


def save_figure(fig: plt.Figure, filename: str, output_dir: str, 
                formats: List[str] = ['pdf', 'png']) -> None:
    """
    Save figure in multiple formats with consistent settings.
    
    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        output_dir: Output directory path
        formats: List of file formats to save
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        filepath = output_path / f"{filename}.{fmt}"
        fig.savefig(filepath, format=fmt, bbox_inches='tight', 
                   dpi=300, transparent=True)
        print(f"✓ Saved: {filepath}")


def create_performance_heatmap(data: pd.DataFrame, title: str, 
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create a performance heatmap for model evaluation metrics.
    
    Args:
        data: DataFrame with performance metrics
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(data, annot=True, cmap='RdYlBu_r', center=0.5,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    return fig


def create_confidence_distribution(confidences: np.ndarray, labels: List[str],
                                 title: str = "Confidence Distribution") -> plt.Figure:
    """
    Create confidence score distribution plots.
    
    Args:
        confidences: Array of confidence scores
        labels: List of class labels
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    # Overall distribution
    axes[0, 0].hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Overall Confidence Distribution')
    axes[0, 0].set_xlabel('Confidence Score')
    axes[0, 0].set_ylabel('Frequency')
    
    # Box plot by class
    if len(labels) <= len(confidences):
        df = pd.DataFrame({'confidence': confidences[:len(labels)], 'class': labels})
        sns.boxplot(data=df, x='class', y='confidence', ax=axes[0, 1])
        axes[0, 1].set_title('Confidence by Class')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Cumulative distribution
    sorted_conf = np.sort(confidences)
    y_vals = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf)
    axes[1, 0].plot(sorted_conf, y_vals, linewidth=2, color='orange')
    axes[1, 0].set_title('Cumulative Distribution')
    axes[1, 0].set_xlabel('Confidence Score')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Violin plot
    axes[1, 1].violinplot([confidences], positions=[1], showmeans=True)
    axes[1, 1].set_title('Confidence Distribution Shape')
    axes[1, 1].set_ylabel('Confidence Score')
    axes[1, 1].set_xticks([1])
    axes[1, 1].set_xticklabels(['All Classes'])
    
    plt.tight_layout()
    return fig


def create_shap_summary_plot(shap_values: np.ndarray, feature_names: List[str],
                           title: str = "SHAP Feature Importance") -> plt.Figure:
    """
    Create SHAP summary visualization.
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    # Feature importance (mean absolute SHAP values)
    importance = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(importance)[-20:]  # Top 20 features
    
    axes[0].barh(range(len(sorted_idx)), importance[sorted_idx], color='steelblue')
    axes[0].set_yticks(range(len(sorted_idx)))
    axes[0].set_yticklabels([feature_names[i] for i in sorted_idx])
    axes[0].set_xlabel('Mean |SHAP value|')
    axes[0].set_title('Top 20 Feature Importance')
    
    # SHAP value distribution
    axes[1].hist(shap_values.flatten(), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('SHAP value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('SHAP Value Distribution')
    axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.8)
    
    plt.tight_layout()
    return fig


def create_attention_heatmap(attention_weights: np.ndarray, tokens: List[str],
                           title: str = "Attention Weights") -> plt.Figure:
    """
    Create attention weight heatmap.
    
    Args:
        attention_weights: Attention weights matrix
        tokens: List of token strings
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(attention_weights, 
                xticklabels=tokens[:attention_weights.shape[1]] if tokens else False,
                yticklabels=tokens[:attention_weights.shape[0]] if tokens else False,
                cmap='Blues', ax=ax, cbar_kws={"shrink": 0.8})
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Token Position')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig


def create_interactive_comparison_chart(data: Dict[str, List[float]], 
                                      title: str = "Performance Comparison") -> go.Figure:
    """
    Create interactive comparison chart using Plotly.
    
    Args:
        data: Dictionary with method names as keys and performance values as lists
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, (method, values) in enumerate(data.items()):
        fig.add_trace(go.Box(
            y=values,
            name=method,
            boxpoints='all',
            marker=dict(color=colors[i % len(colors)]),
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        yaxis_title="Performance Score",
        xaxis_title="Method",
        showlegend=False,
        template="plotly_white",
        height=600
    )
    
    return fig


def create_training_progress_plot(training_history: Dict[str, List[float]],
                                title: str = "Training Progress") -> plt.Figure:
    """
    Create training progress visualization.
    
    Args:
        training_history: Dictionary with 'loss', 'accuracy', 'val_loss', 'val_accuracy' keys
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    epochs = range(1, len(training_history['loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, training_history['loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in training_history:
        axes[0].plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, training_history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in training_history:
        axes[1].plot(epochs, training_history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_component_diagram(components: List[Dict], connections: List[Tuple],
                           title: str = "System Components") -> plt.Figure:
    """
    Create a system component diagram.
    
    Args:
        components: List of component dictionaries with 'name', 'pos', 'color' keys
        connections: List of connection tuples (start_idx, end_idx)
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Draw components
    for i, comp in enumerate(components):
        x, y = comp['pos']
        rect = FancyBboxPatch(
            (x, y), 1.5, 0.8,
            boxstyle="round,pad=0.1",
            facecolor=comp.get('color', COLORS['processing']),
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + 0.75, y + 0.4, comp['name'], ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Draw connections
    for start_idx, end_idx in connections:
        start_pos = components[start_idx]['pos']
        end_pos = components[end_idx]['pos']
        
        ax.annotate('', 
                   xy=(end_pos[0], end_pos[1] + 0.4), 
                   xytext=(start_pos[0] + 1.5, start_pos[1] + 0.4),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_title(title, fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    return fig


# Utility functions for data loading and processing
def load_model_results(results_path: str) -> Dict:
    """Load model evaluation results from JSON file."""
    import json
    with open(results_path, 'r') as f:
        return json.load(f)


def process_shap_data(shap_file_path: str) -> Tuple[np.ndarray, List[str]]:
    """Process SHAP analysis results."""
    # Placeholder - implement based on actual SHAP output format

    """Process SHAP analysis results.

    Assumes the SHAP data is stored as a CSV file where columns are feature names
    and rows are SHAP values for each sample.

    Args:
        shap_file_path: Path to the SHAP values CSV file.

    Returns:
        Tuple of (shap_values: np.ndarray, feature_names: List[str])
    """
    df = pd.read_csv(shap_file_path)
    feature_names = list(df.columns)
    shap_values = df.values
    return shap_values, feature_names
def format_number(num: float, precision: int = 3) -> str:
    """Format numbers for display in plots."""
    if abs(num) >= 1000:
        return f"{num/1000:.1f}K"
    elif abs(num) >= 1:
        return f"{num:.{precision}f}"
    else:
        return f"{num:.{precision}f}"


# Export all functions for easy import
__all__ = [
    'setup_plotting_style', 'create_color_palette', 'save_figure',
    'create_performance_heatmap', 'create_confidence_distribution',
    'create_shap_summary_plot', 'create_attention_heatmap',
    'create_interactive_comparison_chart', 'create_training_progress_plot',
    'create_component_diagram', 'load_model_results', 'process_shap_data',
    'format_number', 'COLORS', 'STYLE_CONFIG'
]
