import shap
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiLabelBERT(nn.Module):
    """Multi-label BERT model for legal clause classification"""
    def __init__(self, model_name, num_labels, dropout=0.3):
        super(MultiLabelBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return {'logits': logits, 'hidden_states': outputs.last_hidden_state}

class LegalModelWrapper:
    """Wrapper for BERT model to work with SHAP for legal clause analysis"""
    
    def __init__(self, model, tokenizer, clause_types, clean_clause_names, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.clause_types = clause_types
        self.clean_clause_names = clean_clause_names
        self.max_length = max_length
        self.device = next(model.parameters()).device
        
    def predict(self, texts):
        """Predict probabilities for a list of texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Get predictions
                outputs = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(outputs['logits']).cpu().numpy()
                predictions.append(probs[0])
        
        return np.array(predictions)
    
    def predict_single_class(self, texts, class_idx):
        """Predict probability for a specific class (for targeted SHAP analysis)"""
        probs = self.predict(texts)
        return probs[:, class_idx]

def load_legal_model():
    """Load the trained legal BERT model and associated data"""
    
    import os
    
    # Get the script directory and construct absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from scripts/
    
    metadata_path = os.path.join(project_root, 'data', 'processed', 'metadata.json')
    training_results_path = os.path.join(project_root, 'models', 'bert', 'training_results.json')
    model_path = os.path.join(project_root, 'models', 'bert', 'final_model.pt')
    tokenizer_path = os.path.join(project_root, 'models', 'bert')
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load training results
    with open(training_results_path, 'r') as f:
        training_results = json.load(f)
    
    # Initialize model
    MODEL_NAME = training_results['model_config']['model_name']
    num_labels = training_results['model_config']['num_labels']
    
    model = MultiLabelBERT(MODEL_NAME, num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    return model, tokenizer, metadata, training_results

def load_legal_data(data_filename='test_multi_label.csv'):
    """Load legal contract data for SHAP analysis"""
    import os
    
    # Get absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'processed', data_filename)
    
    data = pd.read_csv(data_path)
    
    # Parse labels if they're stored as strings
    import ast
    if 'labels' in data.columns:
        data['labels'] = data['labels'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    
    return data

def setup_shap_explainer(model_wrapper, background_texts, background_size=20):
    """Set up SHAP explainer for legal text analysis"""
    
    # Create background dataset (smaller for computational efficiency)
    background_sample = background_texts[:background_size] if len(background_texts) >= background_size else background_texts
    
    print(f"Setting up SHAP explainer with {len(background_sample)} background samples...")
    
    # Create SHAP explainer
    explainer = shap.Explainer(model_wrapper.predict, background_sample)
    
    return explainer

def explain_legal_predictions(explainer, model_wrapper, texts, clause_types, clean_clause_names, top_k=5):
    """Generate SHAP explanations for legal clause predictions"""
    
    explanations = {}
    
    for i, text in enumerate(texts):
        print(f"\nAnalyzing text {i+1}/{len(texts)}: {text[:100]}...")
        
        # Get model predictions
        probs = model_wrapper.predict([text])[0]
        top_classes = np.argsort(probs)[-top_k:][::-1]
        
        #  DEBUG SECTION:
        print(f"  Top {top_k} predictions:")
        for idx in top_classes:
            clause_type = clause_types[idx]
            clean_name = clean_clause_names.get(clause_type, clause_type.split('"')[1] if '"' in clause_type else clause_type[:30])
            print(f"    {clean_name}: {probs[idx]:.4f}")

        text_explanations = {}
        
        try:
            # Generate SHAP values
            shap_values = explainer([text])
            
            for class_idx in top_classes:
                clause_type = clause_types[class_idx]
                clean_name = clean_clause_names.get(clause_type, clause_type.split('"')[1] if '"' in clause_type else clause_type[:30])
                confidence = probs[class_idx]
                
                if confidence > 0.01:  # Lower threshold to capture more predictions
                    try:
                        # Extract SHAP values for this class
                        if hasattr(shap_values, 'values'):
                            class_shap_values = shap_values.values[0][:, class_idx] if shap_values.values[0].ndim > 1 else shap_values.values[0]
                        else:
                            class_shap_values = shap_values[0][:, class_idx] if shap_values[0].ndim > 1 else shap_values[0]
                        
                        text_explanations[clean_name] = {
                            'confidence': confidence,
                            'class_idx': class_idx,
                            'shap_values': class_shap_values,
                            'clause_type': clause_type
                        }
                        
                        print(f"  • {clean_name}: {confidence:.3f}")
                        
                    except Exception as e:
                        print(f"  Warning: Could not extract SHAP values for {clean_name}: {e}")
                        
        except Exception as e:
            print(f"  Error generating SHAP explanations for text {i+1}: {e}")
            
        explanations[f"text_{i+1}"] = {
            'text': text,
            'explanations': text_explanations
        }
    
    return explanations

def plot_legal_shap_summary(explanations, save_path=None):
    """Create summary plots for legal SHAP analysis"""
    
    # Collect all confidence scores by clause type
    clause_confidences = {}
    
    for text_key, text_data in explanations.items():
        for clause_name, explanation in text_data['explanations'].items():
            if clause_name not in clause_confidences:
                clause_confidences[clause_name] = []
            clause_confidences[clause_name].append(explanation['confidence'])
    
    #  DEBUG SECTION:
    print(f"DEBUG: Found {len(clause_confidences)} unique clause types")
    for name, confs in clause_confidences.items():
        print(f"  {name}: {len(confs)} predictions, avg={np.mean(confs):.4f}")
    
    if not clause_confidences:
        print("WARNING: No clause predictions found for plotting!")
        return

    # Plot confidence distribution by clause type
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average confidence by clause type
    clause_names = list(clause_confidences.keys())
    avg_confidences = [np.mean(clause_confidences[name]) for name in clause_names]
    
    bars = ax1.barh(range(len(clause_names)), avg_confidences)
    ax1.set_yticks(range(len(clause_names)))
    ax1.set_yticklabels(clause_names)
    ax1.set_xlabel('Average Confidence')
    ax1.set_title('Average Model Confidence by Clause Type')
    ax1.grid(True, alpha=0.3)
    
    # Color bars by confidence
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(avg_confidences[i] / max(avg_confidences)))
    
    # Confidence distribution
    all_confidences = [conf for confs in clause_confidences.values() for conf in confs]
    ax2.hist(all_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Model Confidence Scores')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to: {save_path}")
    
    plt.show()

def generate_shap_report(explanations, output_path=None):
    """Generate a comprehensive SHAP analysis report for legal documents"""
    
    report = []
    report.append("="*80)
    report.append("SHAP ANALYSIS REPORT - LEGAL CLAUSE DETECTION")
    report.append("="*80)
    
    # Overall statistics
    total_texts = len(explanations)
    total_predictions = sum(len(data['explanations']) for data in explanations.values())
    
    report.append(f"\nANALYSIS OVERVIEW")
    report.append("-" * 40)
    report.append(f"  • Total texts analyzed: {total_texts}")
    report.append(f"  • Total clause predictions explained: {total_predictions}")
    
    # Clause type analysis
    clause_stats = {}
    for text_data in explanations.values():
        for clause_name, explanation in text_data['explanations'].items():
            if clause_name not in clause_stats:
                clause_stats[clause_name] = {'count': 0, 'confidences': []}
            clause_stats[clause_name]['count'] += 1
            clause_stats[clause_name]['confidences'].append(explanation['confidence'])
    
    report.append(f"\nCLAUSE TYPE ANALYSIS")
    report.append("-" * 40)
    sorted_clauses = sorted(clause_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    for clause_name, stats in sorted_clauses[:10]:  # Top 10
        avg_conf = np.mean(stats['confidences'])
        report.append(f"  • {clause_name}")
        report.append(f"    - Occurrences: {stats['count']}")
        report.append(f"    - Avg Confidence: {avg_conf:.3f}")
    
    # Individual text analysis
    report.append(f"\nINDIVIDUAL TEXT ANALYSIS")
    report.append("-" * 40)
    
    for text_key, text_data in explanations.items():
        text_preview = text_data['text'][:150] + "..." if len(text_data['text']) > 150 else text_data['text']
        report.append(f"\n{text_key.upper()}:")
        report.append(f"  Text: {text_preview}")
        report.append(f"  Detected clauses: {len(text_data['explanations'])}")
        
        for clause_name, explanation in text_data['explanations'].items():
            report.append(f"    • {clause_name}: {explanation['confidence']:.3f}")
    
    report.append(f"\nSHAP ANALYSIS COMPLETE")
    report.append("="*80)
    
    # Write to file if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print(f"SHAP report saved to: {output_path}")
    
    # Print to console
    for line in report:
        print(line)

def main():
    """Main function to run SHAP analysis on legal documents"""
    
    print("Starting SHAP Analysis for Legal Clause Detection")
    print("="*60)
    
    # Load model and data
    print("Loading legal model and tokenizer...")
    model, tokenizer, metadata, training_results = load_legal_model()
    
    clause_types = metadata['clause_types']
    clean_clause_names = metadata['clean_clause_names']
    
    print(f"Model loaded: {training_results['model_config']['model_name']}")
    print(f"Number of clause types: {len(clause_types)}")
    
    # Load test data
    print("\nLoading test data...")
    test_data = load_legal_data('test_multi_label.csv')  # Just pass filename
    print(f"Test samples loaded: {len(test_data)}")
    
    # Create model wrapper
    model_wrapper = LegalModelWrapper(model, tokenizer, clause_types, clean_clause_names)
    
    # Sample texts for analysis (adjust sample size as needed)
    sample_size = min(5, len(test_data))  # Analyze 5 samples for demonstration
    sample_indices = np.random.choice(len(test_data), sample_size, replace=False)
    sample_texts = test_data.iloc[sample_indices]['text'].tolist()
    
    print(f"\nAnalyzing {len(sample_texts)} sample texts...")
    
    # Set up SHAP explainer
    explainer = setup_shap_explainer(model_wrapper, sample_texts[:10])  # Use first 10 as background
    
    # Generate explanations
    print("\nGenerating SHAP explanations...")
    explanations = explain_legal_predictions(
        explainer, model_wrapper, sample_texts, clause_types, clean_clause_names
    )
    
    # Get absolute paths for outputs
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    plot_path = os.path.join(project_root, 'models', 'shap_summary_plot.png')
    report_path = os.path.join(project_root, 'models', 'shap_analysis_report.txt')
    
    # Create visualizations
    print("\nCreating SHAP visualizations...")
    plot_legal_shap_summary(explanations, save_path=plot_path)
    
    # Generate report
    print("\nGenerating SHAP analysis report...")
    generate_shap_report(explanations, output_path=report_path)
    
    print("\nSHAP analysis complete!")
    print(f"Check '{os.path.join(project_root, 'models')}' directory for saved outputs.")

    print("\nQUICK MODEL TEST:")
    sample_text = test_data.iloc[0]['text']
    probs = model_wrapper.predict([sample_text])[0]
    print(f"Max confidence: {np.max(probs):.4f}")
    print(f"Mean confidence: {np.mean(probs):.4f}")
    print(f"Predictions > 0.1: {np.sum(probs > 0.1)}")
    print(f"Predictions > 0.05: {np.sum(probs > 0.05)}")
    print(f"Predictions > 0.01: {np.sum(probs > 0.01)}")

if __name__ == "__main__":
    main()