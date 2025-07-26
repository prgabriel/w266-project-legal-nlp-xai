"""
Legal Document Summarization Fine-Tuning Script
T5-based abstractive summarization for legal contracts and clauses
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalSummarizationDataset(Dataset):
    """
    PyTorch Dataset for legal document summarization
    """
    def __init__(self, texts: List[str], summaries: List[str], tokenizer, 
                 max_input_length: int = 1024, max_target_length: int = 256):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        summary = str(self.summaries[idx])
        
        # Add task prefix for T5
        input_text = f"summarize legal document: {text}"
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            summary,
            add_special_tokens=True,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

class LegalSummarizationModel:
    """
    T5-based Legal Document Summarization with Fine-tuning Support
    Optimized for legal contracts, clauses, and legal reasoning
    """
    
    def __init__(self, 
             model_name: str = 't5-base',
             output_dir: str = '../models/t5'):  # Fixed path
        
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Initialize tokenizer and model
        logger.info(f"Loading {model_name} for legal summarization...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Add legal-specific tokens
        self._add_legal_tokens()
        
        # ROUGE scorer for evaluation
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_metrics': [],
            'rouge_scores': []
        }
        
        logger.info(f"Model initialized with {self.model.num_parameters():,} parameters")
    
    def _add_legal_tokens(self):
        """Add legal-specific tokens to the tokenizer"""
        legal_tokens = [
            '<legal_clause>', '<contract_section>', '<party_name>',
            '<date_reference>', '<monetary_amount>', '<legal_term>',
            '<agreement_type>', '<jurisdiction>', '<termination_clause>',
            '<liability_limit>', '<payment_terms>', '<delivery_terms>',
            '<force_majeure>', '<intellectual_property>', '<confidentiality>',
            '<warranty>', '<indemnification>', '<dispute_resolution>'
        ]
        
        # Fix: Use get_vocab() instead of .vocab for T5 tokenizer
        try:
            existing_vocab = self.tokenizer.get_vocab()
            new_tokens = [token for token in legal_tokens if token not in existing_vocab]
            
            if new_tokens:
                self.tokenizer.add_tokens(new_tokens)
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Added {len(new_tokens)} legal-specific tokens: {new_tokens}")
            else:
                logger.info("All legal tokens already exist in tokenizer vocabulary")
                
        except Exception as e:
            logger.error(f"Error adding legal tokens: {e}")
            # Fallback: try to add all tokens without checking
            try:
                num_added = self.tokenizer.add_tokens(legal_tokens)
                if num_added > 0:
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    logger.info(f"Added {num_added} legal tokens (fallback method)")
            except Exception as fallback_error:
                logger.error(f"Fallback token addition also failed: {fallback_error}")
    
    def create_legal_training_data(self) -> Tuple[List[str], List[str]]:
        """
        Create training data from legal contract sections
        This would typically load from your preprocessed legal documents
        """
        # Placeholder - in practice, load from your CUAD or legal document dataset
        sample_texts = [
            """This Agreement shall commence on the Effective Date and shall continue for a period of three (3) years, 
            unless earlier terminated in accordance with the provisions hereof. Either party may terminate this Agreement 
            at any time upon thirty (30) days written notice to the other party. Upon termination, all rights and 
            obligations shall cease except those that by their nature should survive termination.""",
            
            """The Licensee shall pay to the Licensor a royalty equal to five percent (5%) of Net Sales of Licensed Products. 
            Royalty payments shall be made quarterly within forty-five (45) days after the end of each calendar quarter. 
            Each payment shall be accompanied by a written report showing the calculation of royalties due.""",
            
            """Each party acknowledges that it may have access to certain confidential information of the other party. 
            Each party agrees to maintain in confidence all confidential information received from the other party and 
            not to disclose such information to third parties without prior written consent."""
        ]
        
        sample_summaries = [
            "Three-year agreement with 30-day termination notice. Certain obligations survive termination.",
            "5% royalty on net sales, paid quarterly with 45-day reporting requirement.",
            "Mutual confidentiality obligations with 5-year survival period and standard exceptions for public information."
        ]
        
        logger.info(f"Created {len(sample_texts)} sample legal text-summary pairs")
        return sample_texts, sample_summaries
    
    def load_cuad_summaries(self, data_dir: str = '../../data/processed') -> Tuple[List[str], List[str]]:
        """
        Load legal texts and create extractive summaries from CUAD dataset
        """
        texts = []
        summaries = []
        
        # Look for processed CSV files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and not f.startswith('test_')]
        
        for csv_file in csv_files[:5]:  # Limit for demonstration
            try:
                df = pd.read_csv(os.path.join(data_dir, csv_file))
                
                if 'context' in df.columns:
                    for _, row in df.iterrows():
                        context = str(row['context'])
                        
                        # Create extractive summary (first few sentences)
                        sentences = sent_tokenize(context)
                        if len(sentences) >= 3:
                            summary = ' '.join(sentences[:2])  # First 2 sentences
                            
                            # Filter by length
                            if 100 <= len(context) <= 2000 and 20 <= len(summary) <= 300:
                                texts.append(context)
                                summaries.append(summary)
                
            except Exception as e:
                logger.warning(f"Could not process {csv_file}: {e}")
        
        logger.info(f"Loaded {len(texts)} legal text-summary pairs from CUAD data")
        return texts, summaries
    
    def load_cuad_summaries_enhanced(self, data_dir: str = None, 
                                   max_samples: int = 10000) -> Tuple[List[str], List[str]]:
        """
        Enhanced CUAD data loading with proper summarization approach
        """
        # Determine the correct data directory path
        if data_dir is None:
            # Get the script directory and navigate to data/processed
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            data_dir = os.path.join(project_root, 'data', 'processed')
        
        logger.info(f"Looking for data in: {data_dir}")
        
        if not os.path.exists(data_dir):
            logger.error(f"Data directory not found: {data_dir}")
            return [], []
        
        texts = []
        summaries = []
        
        # Load from question-answering dataset for context-answer pairs
        qa_file = os.path.join(data_dir, 'train_question_answering.csv')
        if os.path.exists(qa_file):
            logger.info(f"Loading Q&A data from {qa_file}...")
            try:
                df = pd.read_csv(qa_file)
                logger.info(f"Found {len(df)} Q&A samples")
                
                # Create summarization pairs from Q&A data
                for _, row in df.iterrows():
                    if len(texts) >= max_samples:
                        break
                        
                    context = str(row['context']).strip()
                    question = str(row['question']).strip()
                    answer = str(row['answer_text']).strip() if pd.notna(row['answer_text']) else ""
                    
                    # Filter by length and quality
                    if (200 <= len(context) <= 2000 and 
                        len(question) > 10 and 
                        len(answer) > 5 and
                        answer.lower() != 'nan'):
                        
                        # Create summary from question + answer
                        summary = f"Question: {question} Answer: {answer}"
                        
                        # Ensure summary is reasonable length
                        if 30 <= len(summary) <= 400:
                            texts.append(context)
                            summaries.append(summary)
                
                logger.info(f"Loaded {len(texts)} Q&A pairs for summarization")
                
            except Exception as e:
                logger.error(f"Error loading Q&A data: {e}")
        else:
            logger.warning(f"Q&A file not found: {qa_file}")
        
        # Also load from binary classification files for additional context
        try:
            binary_files = [f for f in os.listdir(data_dir) 
                           if f.startswith('train_binary_') and f.endswith('.csv')]
            logger.info(f"Found {len(binary_files)} binary classification files")
        except FileNotFoundError:
            logger.error(f"Data directory not accessible: {data_dir}")
            binary_files = []
        
        for binary_file in binary_files[:10]:  # Limit to prevent overwhelming
            if len(texts) >= max_samples:
                break
                
            try:
                file_path = os.path.join(data_dir, binary_file)
                df = pd.read_csv(file_path)
                logger.info(f"Processing {binary_file}: {len(df)} samples")
                
                for _, row in df.iterrows():
                    if len(texts) >= max_samples:
                        break
                        
                    if 'text' in df.columns and 'clause_type' in df.columns:
                        text = str(row['text']).strip()
                        clause_type = str(row['clause_type']).strip()
                        
                        # Create extractive summary
                        sentences = sent_tokenize(text)
                        if len(sentences) >= 2 and 200 <= len(text) <= 2000:
                            # Use first 1-2 sentences + clause type info
                            clause_name = "Legal Clause"
                            if 'related to' in clause_type:
                                try:
                                    clause_name = clause_type.split('related to')[1].split('that should')[0].strip().strip('"')
                                except:
                                    clause_name = "Legal Clause"
                            
                            summary = f"{sentences[0]} (Type: {clause_name})"
                            
                            if 30 <= len(summary) <= 400:
                                texts.append(text)
                                summaries.append(summary)
                        
            except Exception as e:
                logger.warning(f"Could not process {binary_file}: {e}")
        
        logger.info(f"Loaded {len(texts)} enhanced legal text-summary pairs")
        return texts, summaries

    def create_legal_training_data_enhanced(self) -> Tuple[List[str], List[str]]:
        """
        Create larger, more realistic training dataset
        """
        # First try to load from CUAD data
        try:
            texts, summaries = self.load_cuad_summaries_enhanced(max_samples=5000)
            if len(texts) >= 100:  # Good amount of data
                logger.info(f"Successfully loaded {len(texts)} samples from CUAD data")
                return texts, summaries
            else:
                logger.warning(f"Only found {len(texts)} CUAD samples, falling back to synthetic data")
        except Exception as e:
            logger.warning(f"Could not load CUAD data: {e}")
        
        # Fallback: create more diverse sample data
        logger.info("Generating synthetic legal training data...")
        sample_texts = [
            """This Software License Agreement ("Agreement") is entered into on January 15, 2024, 
            between TechCorp Inc., a Delaware corporation ("Licensor"), and BusinessSoft LLC, 
            a California limited liability company ("Licensee"). The Licensor hereby grants to 
            the Licensee a non-exclusive, non-transferable license to use the software described 
            in Exhibit A for internal business purposes only. The license term shall be for 
            three (3) years from the effective date, with automatic renewal for successive 
            one-year periods unless either party provides sixty (60) days written notice of 
            non-renewal. The Licensee agrees to pay an annual license fee of $50,000, payable 
            in advance on each anniversary of the effective date.""",
            
            """The Licensee shall pay to the Licensor a royalty equal to five percent (5%) of Net Sales of Licensed Products. 
            Royalty payments shall be made quarterly within forty-five (45) days after the end of each calendar quarter. 
            Each payment shall be accompanied by a written report showing the calculation of royalties due. 
            The Licensor reserves the right to audit the Licensee's records relating to Net Sales upon reasonable notice 
            and during normal business hours. Any discrepancies found during such audit shall be resolved promptly.""",
            
            """Each party acknowledges that it may have access to certain confidential information of the other party. 
            Each party agrees to maintain in confidence all confidential information received from the other party and 
            not to disclose such information to third parties without prior written consent. This obligation shall 
            survive termination of this Agreement for a period of five (5) years. Confidential information does not 
            include information that is publicly available or independently developed.""",
            
            """This Agreement shall commence on the Effective Date and shall continue for a period of three (3) years, 
            unless earlier terminated in accordance with the provisions hereof. Either party may terminate this Agreement 
            at any time upon thirty (30) days written notice to the other party. Upon termination, all rights and 
            obligations shall cease except those that by their nature should survive termination, including but not 
            limited to confidentiality obligations, payment obligations, and limitation of liability provisions.""",
            
            """The parties acknowledge that this Agreement shall be governed by and construed in accordance with the 
            laws of the State of Delaware, without regard to its conflict of laws principles. Any disputes arising 
            under this Agreement shall be resolved through binding arbitration in accordance with the rules of the 
            American Arbitration Association. The arbitration shall take place in Wilmington, Delaware, and the 
            decision of the arbitrator shall be final and binding upon both parties."""
        ]
        
        sample_summaries = [
            "Software license agreement between TechCorp and BusinessSoft for 3-year non-exclusive license with $50K annual fee and auto-renewal.",
            "5% royalty on net sales, paid quarterly with 45-day reporting requirement and audit rights reserved by licensor.",
            "Mutual confidentiality obligations with 5-year survival period and standard exceptions for public information.",
            "Three-year agreement with 30-day termination notice and surviving obligations for confidentiality and payments.",
            "Delaware law governs with binding arbitration in Wilmington for dispute resolution through AAA rules."
        ]
        
        # Replicate with variations to create more training data
        expanded_texts = []
        expanded_summaries = []
        
        for i in range(500):  # Create 500 samples for better training
            idx = i % len(sample_texts)
            text = sample_texts[idx]
            summary = sample_summaries[idx]
            
            # Add slight variations
            if i >= len(sample_texts):
                # Vary amounts, dates, company names, etc.
                text = re.sub(r'\$[\d,]+', f'${(i*1000+25000):,}', text)
                text = re.sub(r'\d+ \(\d+\) years?', f'{(i%5)+1} ({(i%5)+1}) years', text)
                text = re.sub(r'TechCorp Inc\.', f'Company{i%10} Inc.', text)
                text = re.sub(r'BusinessSoft LLC', f'Business{i%8} LLC', text)
                
            expanded_texts.append(text)
            expanded_summaries.append(summary)
        
        logger.info(f"Created {len(expanded_texts)} enhanced sample legal text-summary pairs")
        return expanded_texts, expanded_summaries
    
    def preprocess_legal_text(self, text: str) -> str:
        """
        Preprocess legal text for better summarization
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Legal-specific preprocessing
        # Normalize common legal phrases
        legal_normalizations = {
            r'\b(shall|will)\b': 'must',
            r'\bhereof\b': 'of this agreement',
            r'\bherein\b': 'in this agreement',
            r'\bwhereas\b': 'given that',
            r'\bthereof\b': 'of that'
        }
        
        import re
        for pattern, replacement in legal_normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def summarize(self, 
                  text: str, 
                  max_length: int = 200, 
                  min_length: int = 50, 
                  num_beams: int = 4,
                  do_sample: bool = False,
                  temperature: float = 1.0,
                  length_penalty: float = 1.0) -> Dict[str, Union[str, float]]:
        """
        Generate abstractive summary for legal text
        """
        # Preprocess input
        processed_text = self.preprocess_legal_text(text)
        input_text = f"summarize legal document: {processed_text}"
        
        # Tokenize
        inputs = self.tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True
        )
        
        # Generate summary
        self.model.eval()
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                length_penalty=length_penalty,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Calculate summary statistics
        original_words = len(word_tokenize(text))
        summary_words = len(word_tokenize(summary))
        compression_ratio = summary_words / original_words if original_words > 0 else 0
        
        return {
            'summary': summary,
            'original_length': original_words,
            'summary_length': summary_words,
            'compression_ratio': compression_ratio
        }
    
    def batch_summarize(self, texts: List[str], **kwargs) -> List[Dict]:
        """Summarize multiple texts efficiently"""
        results = []
        
        logger.info(f"Summarizing {len(texts)} legal documents...")
        
        for i, text in enumerate(texts):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(texts)}")
            
            result = self.summarize(text, **kwargs)
            results.append(result)
        
        return results
    
    def compute_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores for evaluation"""
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_1_scores.append(scores['rouge1'].fmeasure)
            rouge_2_scores.append(scores['rouge2'].fmeasure)
            rouge_l_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge_1_scores),
            'rouge2': np.mean(rouge_2_scores),
            'rougeL': np.mean(rouge_l_scores),
            'rouge1_std': np.std(rouge_1_scores),
            'rouge2_std': np.std(rouge_2_scores),
            'rougeL_std': np.std(rouge_l_scores)
        }
    
    def evaluate_model(self, test_texts: List[str], reference_summaries: List[str]) -> Dict:
        """
        Comprehensive evaluation of summarization model
        """
        logger.info("Evaluating summarization model...")
        
        # Generate summaries
        generated_summaries = []
        compression_ratios = []
        
        for text in test_texts:
            result = self.summarize(text)
            generated_summaries.append(result['summary'])
            compression_ratios.append(result['compression_ratio'])
        
        # Compute ROUGE scores
        rouge_scores = self.compute_rouge_scores(generated_summaries, reference_summaries)
        
        # Additional metrics
        avg_compression_ratio = np.mean(compression_ratios)
        
        evaluation_results = {
            'rouge_scores': rouge_scores,
            'avg_compression_ratio': avg_compression_ratio,
            'num_test_samples': len(test_texts),
            'sample_summaries': [
                {
                    'original': test_texts[i][:200] + '...' if len(test_texts[i]) > 200 else test_texts[i],
                    'reference': reference_summaries[i],
                    'generated': generated_summaries[i],
                    'compression_ratio': compression_ratios[i]
                }
                for i in range(min(3, len(test_texts)))  # Show first 3 examples
            ]
        }
        
        # Log results
        logger.info(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
        logger.info(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
        logger.info(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        logger.info(f"Avg Compression Ratio: {avg_compression_ratio:.4f}")
        
        return evaluation_results
    
    def fine_tune(self,
                  train_texts: List[str],
                  train_summaries: List[str],
                  val_texts: Optional[List[str]] = None,
                  val_summaries: Optional[List[str]] = None,
                  epochs: int = 3,
                  batch_size: int = 4,
                  learning_rate: float = 5e-5,
                  warmup_steps: int = 100,
                  weight_decay: float = 0.01,
                  early_stopping_patience: int = 2):
        """
        Fine-tune T5 model on legal summarization task
        """
        logger.info("Starting fine-tuning for legal summarization...")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create datasets
        train_dataset = LegalSummarizationDataset(
            train_texts, train_summaries, self.tokenizer
        )
        
        val_dataset = None
        if val_texts and val_summaries:
            val_dataset = LegalSummarizationDataset(
                val_texts, val_summaries, self.tokenizer
            )
        
        # Training arguments optimized for summarization
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=os.path.join(self.output_dir, 'logs'),
            logging_steps=50,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            report_to=None,
            gradient_accumulation_steps=2,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=2,
            remove_unused_columns=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if val_dataset else []
        )
        
        # Fine-tune
        train_result = trainer.train()
        
        # Save training history
        self.training_history['train_loss'] = [log['train_loss'] for log in trainer.state.log_history if 'train_loss' in log]
        if val_dataset:
            self.training_history['val_metrics'] = [log for log in trainer.state.log_history if 'eval_loss' in log]
        
        # Evaluate with ROUGE scores if validation data available
        if val_texts and val_summaries:
            logger.info("Computing ROUGE scores on validation set...")
            val_results = self.evaluate_model(val_texts, val_summaries)
            self.training_history['rouge_scores'] = val_results['rouge_scores']
        
        # Save training results
        training_results = {
            'training_history': self.training_history,
            'final_train_loss': train_result.training_loss,
            'model_name': self.model_name,
            'training_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_train_samples': len(train_texts)
            }
        }
        
        with open(os.path.join(self.output_dir, 'training_results.json'), 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info(f"Fine-tuning completed. Final loss: {train_result.training_loss:.4f}")
        return trainer
    
    def save_model(self, save_tokenizer: bool = True):
        """Save the fine-tuned model and tokenizer"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(self.output_dir)
        logger.info(f"Model saved to {self.output_dir}")
        
        if save_tokenizer:
            self.tokenizer.save_pretrained(self.output_dir)
            logger.info(f"Tokenizer saved to {self.output_dir}")
        
        # Save model info
        model_info = {
            'model_name': self.model_name,
            'model_type': 'T5 for Legal Document Summarization',
            'task': 'abstractive_summarization',
            'domain': 'legal',
            'num_parameters': self.model.num_parameters()
        }
        
        with open(os.path.join(self.output_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
    
    @classmethod
    def load_pretrained(cls, model_path: str):
        """Load a fine-tuned model"""
        instance = cls.__new__(cls)
        instance.model_name = model_path
        instance.output_dir = model_path
        
        logger.info(f"Loading fine-tuned model from {model_path}")
        instance.tokenizer = T5Tokenizer.from_pretrained(model_path)
        instance.model = T5ForConditionalGeneration.from_pretrained(model_path)
        instance.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        return instance
    
    def generate_legal_summary_report(self, texts: List[str], output_file: str = None) -> Dict:
        """
        Generate comprehensive summarization report for legal documents
        """
        logger.info(f"Generating summary report for {len(texts)} legal documents...")
        
        summaries = self.batch_summarize(texts)
        
        # Aggregate statistics
        total_original_words = sum(s['original_length'] for s in summaries)
        total_summary_words = sum(s['summary_length'] for s in summaries)
        avg_compression_ratio = np.mean([s['compression_ratio'] for s in summaries])
        
        # Create report
        report = {
            'summary_statistics': {
                'num_documents': len(texts),
                'total_original_words': total_original_words,
                'total_summary_words': total_summary_words,
                'overall_compression_ratio': total_summary_words / total_original_words,
                'avg_compression_ratio': avg_compression_ratio,
                'compression_ratio_std': np.std([s['compression_ratio'] for s in summaries])
            },
            'summaries': [
                {
                    'document_id': i,
                    'original_text': text[:200] + '...' if len(text) > 200 else text,
                    'summary': summary['summary'],
                    'compression_ratio': summary['compression_ratio'],
                    'original_length': summary['original_length'],
                    'summary_length': summary['summary_length']
                }
                for i, (text, summary) in enumerate(zip(texts, summaries))
            ],
            'model_info': {
                'model_name': self.model_name,
                'model_type': 'T5 Legal Summarization'
            }
        }
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Summary report saved to {output_file}")
        
        return report


def main():
    """
    Enhanced main pipeline for legal document summarization fine-tuning
    """
    logger.info("Starting Enhanced Legal Document Summarization Pipeline")
    
    # Initialize model
    summarizer = LegalSummarizationModel(model_name='t5-base')
    
    try:
        # Load training data - try enhanced version first
        train_texts, train_summaries = summarizer.create_legal_training_data_enhanced()
        
        logger.info(f"Total dataset size: {len(train_texts)} samples")
        
        if len(train_texts) < 100:
            logger.error("Insufficient training data! Need at least 100 samples for meaningful training.")
            logger.info("Falling back to sample data generation...")
            # Still continue with what we have for demonstration
        
        # Split into train/validation with better ratios
        split_idx = int(0.85 * len(train_texts))  # 85/15 split for larger datasets
        train_texts_split, val_texts = train_texts[:split_idx], train_texts[split_idx:]
        train_summaries_split, val_summaries = train_summaries[:split_idx], train_summaries[split_idx:]
        
        logger.info(f"Training samples: {len(train_texts_split)}, Validation samples: {len(val_texts)}")
        
        # Enhanced training parameters for larger dataset
        # Enable early stopping to prevent overfitting with increased epochs
        early_stopping_patience = 2
        if val_texts and len(val_texts) < 20:
            logger.warning("Validation set is small; early stopping may not be reliable.")

        trainer = summarizer.fine_tune(
            train_texts=train_texts_split,
            train_summaries=train_summaries_split,
            val_texts=val_texts,
            val_summaries=val_summaries,
            epochs=15,  # More epochs for larger dataset
            batch_size=4,  # Larger batch size
            learning_rate=1e-5,  # Slightly lower LR for stability
            warmup_steps=min(max(len(train_texts_split) // 10, 100), 1000),  # 10% warmup, bounded between 100 and 1000
            early_stopping_patience=early_stopping_patience
        )
        
        # Evaluate
        if val_texts:
            evaluation_results = summarizer.evaluate_model(val_texts, val_summaries)
            logger.info("Enhanced evaluation completed!")
            logger.info(f"Validation samples: {len(val_texts)}")
        
        # Save model
        summarizer.save_model()
        
        # Demo with same text
        demo_text = """
        This Software License Agreement ("Agreement") is entered into on January 15, 2024, 
        between TechCorp Inc., a Delaware corporation ("Licensor"), and BusinessSoft LLC, 
        a California limited liability company ("Licensee"). The Licensor hereby grants to 
        the Licensee a non-exclusive, non-transferable license to use the software described 
        in Exhibit A for internal business purposes only. The license term shall be for 
        three (3) years from the effective date, with automatic renewal for successive 
        one-year periods unless either party provides sixty (60) days written notice of 
        non-renewal. The Licensee agrees to pay an annual license fee of $50,000, payable 
        in advance on each anniversary of the effective date.
        """
        
        demo_result = summarizer.summarize(demo_text)
        logger.info(f"\nEnhanced Legal Document Summarization:")
        logger.info(f"Original length: {demo_result['original_length']} words")
        logger.info(f"Summary length: {demo_result['summary_length']} words")
        logger.info(f"Compression ratio: {demo_result['compression_ratio']:.2f}")
        logger.info(f"Summary: {demo_result['summary']}")
        
        logger.info("Enhanced legal summarization pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Enhanced pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()