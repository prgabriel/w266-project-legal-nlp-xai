"""
Legal Clause Extraction Fine-Tuning Script
Multi-label BERT classification for CUAD legal clause detection
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, classification_report, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalClauseDataset(Dataset):
    """
    PyTorch Dataset for legal clause multi-label classification
    """
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = torch.FloatTensor(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

class LegalClauseExtractor:
    """
    Multi-label BERT fine-tuning for legal clause detection
    Supports 41 CUAD clause types with clean name mapping
    """
    
    def __init__(self, 
                 model_name: str = 'bert-base-uncased',
                 data_dir: str = 'c:/Users/pgabriel/Documents/Berkeley/w266-project-legal-nlp-xai/data/processed',
                 output_dir: str = 'c:/Users/pgabriel/Documents/Berkeley/w266-project-legal-nlp-xai/models/bert'):
        
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Load metadata with clean clause names
        self.metadata = self._load_metadata()
        self.clause_types = self.metadata['clause_types']
        self.clean_clause_names = self.metadata['clean_clause_names']
        self.num_labels = len(self.clause_types)
        
        logger.info(f"Initialized LegalClauseExtractor with {self.num_labels} clause types")
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        )
        
        # Training metrics storage
        self.training_history = {
            'train_loss': [],
            'val_metrics': []
        }
    
    def _load_metadata(self) -> Dict:
        """Load clause metadata with clean names mapping"""
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded metadata for {len(metadata['clause_types'])} clause types")
        return metadata
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed CUAD data from CSV files
        Returns train, validation, and test DataFrames
        """
        train_files = []
        test_files = []
        
        # Find all processed CSV files
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                if filename.startswith('train_'):
                    train_files.append(filename)
                elif filename.startswith('test_'):
                    test_files.append(filename)
        
        if not train_files or not test_files:
            raise FileNotFoundError("No processed train/test CSV files found in data directory")
        
        # Load and combine datasets
        train_dfs = []
        test_dfs = []
        
        for train_file in train_files:
            df = pd.read_csv(os.path.join(self.data_dir, train_file))
            clause_type = train_file.replace('train_binary_', '').replace('.csv', '')
            df['clause_type'] = clause_type
            train_dfs.append(df)
        
        for test_file in test_files:
            df = pd.read_csv(os.path.join(self.data_dir, test_file))
            clause_type = test_file.replace('test_binary_', '').replace('.csv', '')
            df['clause_type'] = clause_type
            test_dfs.append(df)
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        # Create multi-label format
        train_df = self._convert_to_multilabel_format(train_df)
        test_df = self._convert_to_multilabel_format(test_df)
        
        # Split train into train/val
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
        
        logger.info(f"Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
        return train_df, val_df, test_df
    
    def _convert_to_multilabel_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert individual clause CSV files to multi-label format
        Groups by context and creates label vectors
        """
        # Group by context to create multi-label examples
        grouped = df.groupby('context').agg({
            'clause_type': list,
            'label': list
        }).reset_index()
        
        # Create multi-label vectors
        multilabel_data = []
        for _, row in grouped.iterrows():
            context = row['context']
            clause_types = row['clause_type'] 
            labels = row['label']
            
            # Create binary label vector for all 41 clause types
            label_vector = np.zeros(self.num_labels, dtype=int)
            
            for clause_type, label in zip(clause_types, labels):
                if clause_type in self.clause_types:
                    clause_idx = self.clause_types.index(clause_type)
                    label_vector[clause_idx] = int(label)
            
            multilabel_data.append({
                'context': context,
                'labels': label_vector,
                'num_positive_labels': np.sum(label_vector)
            })
        
        return pd.DataFrame(multilabel_data)
    
    def create_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                       test_df: pd.DataFrame, max_length: int = 512) -> Tuple[LegalClauseDataset, LegalClauseDataset, LegalClauseDataset]:
        """Create PyTorch datasets for training"""
        
        train_dataset = LegalClauseDataset(
            texts=train_df['context'].tolist(),
            labels=np.array(train_df['labels'].tolist()),
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        val_dataset = LegalClauseDataset(
            texts=val_df['context'].tolist(),
            labels=np.array(val_df['labels'].tolist()),
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        test_dataset = LegalClauseDataset(
            texts=test_df['context'].tolist(),
            labels=np.array(test_df['labels'].tolist()),
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute multi-label classification metrics"""
        predictions, labels = eval_pred
        
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(torch.tensor(predictions))
        
        # Convert to binary predictions using 0.5 threshold
        binary_predictions = (predictions > 0.5).int().numpy()
        
        # Calculate metrics
        f1_micro = f1_score(labels, binary_predictions, average='micro')
        f1_macro = f1_score(labels, binary_predictions, average='macro')
        f1_weighted = f1_score(labels, binary_predictions, average='weighted')
        
        # Per-clause F1 scores
        per_clause_f1 = f1_score(labels, binary_predictions, average=None)
        
        # Hamming loss (fraction of wrong labels)
        hamming_loss = np.mean(labels != binary_predictions)
        
        metrics = {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'hamming_loss': hamming_loss,
            'per_clause_f1_mean': np.mean(per_clause_f1)
        }
        
        return metrics
    
    def train(self, 
              train_dataset: LegalClauseDataset,
              val_dataset: LegalClauseDataset,
              num_epochs: int = 3,
              batch_size: int = 8,
              learning_rate: float = 2e-5,
              warmup_steps: int = 500,
              weight_decay: float = 0.01,
              early_stopping_patience: int = 2):
        """
        Fine-tune BERT for multi-label legal clause classification
        """
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Training arguments optimized for legal text
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=os.path.join(self.output_dir, 'logs'),
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1_micro",
            greater_is_better=True,
            report_to=None,  # Disable wandb
            gradient_accumulation_steps=2,  # For larger effective batch size
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            dataloader_num_workers=4,
            remove_unused_columns=False
        )
        
        # Create trainer with early stopping
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
        
        logger.info("Starting training...")
        
        # Train the model
        train_result = trainer.train()
        
        # Save training history
        self.training_history['train_loss'] = [log['train_loss'] for log in trainer.state.log_history if 'train_loss' in log]
        self.training_history['val_metrics'] = [log for log in trainer.state.log_history if 'eval_f1_micro' in log]
        
        # Save training results
        with open(os.path.join(self.output_dir, 'training_results.json'), 'w') as f:
            json.dump({
                'training_history': self.training_history,
                'final_train_loss': train_result.training_loss,
                'clause_types': self.clause_types,
                'clean_clause_names': self.clean_clause_names
            }, f, indent=2)
        
        logger.info(f"Training completed. Final loss: {train_result.training_loss:.4f}")
        
        return trainer
    
    def evaluate(self, trainer: Trainer, test_dataset: LegalClauseDataset) -> Dict:
        """
        Comprehensive evaluation on test set
        """
        logger.info("Evaluating on test set...")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        test_predictions = torch.sigmoid(torch.tensor(predictions.predictions))
        binary_predictions = (test_predictions > 0.5).int().numpy()
        test_labels = predictions.label_ids
        
        # Overall metrics
        f1_micro = f1_score(test_labels, binary_predictions, average='micro')
        f1_macro = f1_score(test_labels, binary_predictions, average='macro') 
        f1_weighted = f1_score(test_labels, binary_predictions, average='weighted')
        hamming_loss = np.mean(test_labels != binary_predictions)
        
        # Per-clause analysis with clean names
        per_clause_f1 = f1_score(test_labels, binary_predictions, average=None)
        per_clause_results = []
        
        for i, (clause_type, f1) in enumerate(zip(self.clause_types, per_clause_f1)):
            clean_name = self.clean_clause_names.get(clause_type, clause_type[:50])
            
            # Calculate support (number of positive examples)
            support = np.sum(test_labels[:, i])
            
            per_clause_results.append({
                'clause_type': clause_type,
                'clean_name': clean_name,
                'f1_score': float(f1),
                'support': int(support)
            })
        
        # Sort by F1 score for analysis
        per_clause_results.sort(key=lambda x: x['f1_score'], reverse=True)
        
        evaluation_results = {
            'overall_metrics': {
                'f1_micro': float(f1_micro),
                'f1_macro': float(f1_macro),
                'f1_weighted': float(f1_weighted),
                'hamming_loss': float(hamming_loss)
            },
            'per_clause_results': per_clause_results,
            'num_test_samples': len(test_labels),
            'num_clause_types': self.num_labels
        }
        
        # Save evaluation results
        with open(os.path.join(self.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Print summary
        logger.info(f"Test Results:")
        logger.info(f"  F1 Micro: {f1_micro:.4f}")
        logger.info(f"  F1 Macro: {f1_macro:.4f}")
        logger.info(f"  F1 Weighted: {f1_weighted:.4f}")
        logger.info(f"  Hamming Loss: {hamming_loss:.4f}")
        
        # Top and bottom performing clauses
        logger.info("\nTop 5 performing clauses:")
        for result in per_clause_results[:5]:
            logger.info(f"  {result['clean_name']}: F1={result['f1_score']:.3f} (support={result['support']})")
        
        logger.info("\nBottom 5 performing clauses:")
        for result in per_clause_results[-5:]:
            logger.info(f"  {result['clean_name']}: F1={result['f1_score']:.3f} (support={result['support']})")
        
        return evaluation_results
    
    def save_model(self, save_tokenizer: bool = True):
        """Save the fine-tuned model and tokenizer"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(self.output_dir)
        logger.info(f"Model saved to {self.output_dir}")
        
        if save_tokenizer:
            self.tokenizer.save_pretrained(self.output_dir)
            logger.info(f"Tokenizer saved to {self.output_dir}")
        
        # Save additional metadata
        model_info = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'clause_types': self.clause_types,
            'clean_clause_names': self.clean_clause_names,
            'model_architecture': 'BERT for multi-label legal clause classification'
        }
        
        with open(os.path.join(self.output_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def predict(self, texts: List[str], threshold: float = 0.5) -> List[Dict]:
        """
        Make predictions on new legal texts
        Returns list of predictions with clean clause names
        """
        self.model.eval()
        
        predictions = []
        
        for text in texts:
            # Tokenize input
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**encoding)
                probabilities = torch.sigmoid(outputs.logits).squeeze().numpy()
            
            # Apply threshold and get predicted clauses
            predicted_clauses = []
            for i, (clause_type, prob) in enumerate(zip(self.clause_types, probabilities)):
                if prob > threshold:
                    clean_name = self.clean_clause_names.get(clause_type, clause_type[:50])
                    predicted_clauses.append({
                        'clause_type': clause_type,
                        'clean_name': clean_name,
                        'probability': float(prob)
                    })
            
            # Sort by probability
            predicted_clauses.sort(key=lambda x: x['probability'], reverse=True)
            
            predictions.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'predicted_clauses': predicted_clauses,
                'num_predicted_clauses': len(predicted_clauses)
            })
        
        return predictions


def main():
    """
    Main training pipeline for legal clause extraction
    """
    logger.info("Starting Legal Clause Extraction Fine-Tuning")
    
    # Initialize extractor
    extractor = LegalClauseExtractor()
    
    try:
        # Load processed data
        train_df, val_df, test_df = extractor.load_processed_data()
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = extractor.create_datasets(train_df, val_df, test_df)
        
        # Train model
        trainer = extractor.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=3,
            batch_size=8,
            learning_rate=2e-5
        )
        
        # Evaluate on test set
        evaluation_results = extractor.evaluate(trainer, test_dataset)
        
        # Save model
        extractor.save_model()
        
        logger.info("Training pipeline completed successfully!")
        
        # Demo prediction
        demo_text = """
        This Agreement shall commence on January 1, 2024 and shall continue for a period of three years.
        The Licensee shall not assign this Agreement without prior written consent of the Licensor.
        """
        
        predictions = extractor.predict([demo_text])
        logger.info(f"\nDemo prediction results:")
        for pred in predictions:
            logger.info(f"Text: {pred['text']}")
            logger.info(f"Predicted clauses ({pred['num_predicted_clauses']}):")
            for clause in pred['predicted_clauses']:
                logger.info(f"  - {clause['clean_name']}: {clause['probability']:.3f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()