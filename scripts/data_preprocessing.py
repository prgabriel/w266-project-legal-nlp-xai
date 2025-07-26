# scripts/data_preprocessing.py

import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import os
from typing import Dict, List, Tuple, Any

def extract_clause_name(clause_type: str) -> str:
    """
    Extract clean clause name from verbose CUAD question format.
    
    Examples:
    - 'Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer...' 
      -> 'Agreement Date'
    - 'Highlight the parts (if any) of this contract related to "Anti-Assignment" that should be reviewed...'
      -> 'Anti-Assignment'
    """
    # Try to extract text between first set of quotes
    quote_match = re.search(r'"([^"]*)"', clause_type)
    if quote_match:
        return quote_match.group(1)
    
    # Fallback: extract first few words after "related to"
    related_match = re.search(r'related to ([^"]*?) that should', clause_type)
    if related_match:
        return related_match.group(1).strip()
    
    # Final fallback: take first 50 characters
    return clause_type[:50].strip()

class CUADPreprocessor:
    """
    Comprehensive preprocessing pipeline for CUAD dataset
    """
    
    def __init__(self, raw_data_path: str, output_dir: str = '../data/processed/'):
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir
        self.clause_types = []
        self.clean_clause_names = {}  # Map from verbose question to clean name
        self.mlb = MultiLabelBinarizer()
        
    def load_cuad_data(self) -> Dict[str, Any]:
        """Load CUAD JSON data"""
        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            cuad_data = json.load(f)
        return cuad_data
    
    def parse_cuad_to_dataframe(self, cuad_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Parse CUAD JSON data into structured DataFrame
        Based on your data exploration findings
        """
        rows = []
        
        for doc in cuad_data['data']:
            doc_title = doc['title']
            
            for paragraph in doc['paragraphs']:
                context = paragraph['context']
                
                # Group questions by context to create multi-label examples
                context_clauses = {}
                context_answers = {}
                
                for qa in paragraph['qas']:
                    question = qa['question']  # Clause type
                    question_id = qa['id']
                    is_impossible = qa['is_impossible']
                    
                    context_clauses[question] = not is_impossible
                    
                    # Extract answer spans
                    if not is_impossible and 'answers' in qa:
                        context_answers[question] = [
                            {
                                'text': answer['text'],
                                'start': answer['answer_start']
                            } for answer in qa['answers']
                        ]
                    else:
                        context_answers[question] = []
                
                rows.append({
                    'document_title': doc_title,
                    'context': context,
                    'context_length': len(context),
                    'clause_labels': context_clauses,
                    'clause_answers': context_answers,
                    'num_positive_clauses': sum(context_clauses.values())
                })
        
        return pd.DataFrame(rows)
    
    def clean_text(self, text: str) -> str:
        """Clean legal text while preserving important formatting"""
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove non-printable characters except common legal symbols
        text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
        
        # Normalize common legal abbreviations
        legal_abbrevs = {
            r'\bInc\.\b': 'Incorporated',
            r'\bLLC\b': 'Limited Liability Company',
            r'\bCorp\.\b': 'Corporation',
            r'\b&\b': 'and'
        }
        
        for abbrev, full_form in legal_abbrevs.items():
            text = re.sub(abbrev, full_form, text, flags=re.IGNORECASE)
            
        return text.strip()
    
    def handle_long_contexts(self, df: pd.DataFrame, max_length: int = 4000) -> pd.DataFrame:
        """
        Handle contexts longer than max_length using sliding window
        Based on your finding of ~4000-5000 char contexts
        """
        processed_rows = []
        
        for _, row in df.iterrows():
            context = row['context']
            
            if len(context) <= max_length:
                processed_rows.append(row)
            else:
                # Split long contexts with overlap
                window_size = max_length
                overlap = max_length // 4  # 25% overlap
                
                start = 0
                window_count = 0
                
                while start < len(context):
                    end = min(start + window_size, len(context))
                    window_text = context[start:end]
                    
                    # Create new row for this window
                    new_row = row.copy()
                    new_row['context'] = window_text
                    new_row['context_length'] = len(window_text)
                    new_row['window_id'] = f"{row.name}_{window_count}"
                    new_row['is_windowed'] = True
                    
                    # Adjust answer positions for windowed contexts
                    adjusted_answers = {}
                    for clause_type, answers in row['clause_answers'].items():
                        adjusted_clause_answers = []
                        for answer in answers:
                            answer_start = answer['start']
                            answer_end = answer_start + len(answer['text'])
                            
                            # Check if answer is within this window
                            if answer_start >= start and answer_end <= end:
                                adjusted_answer = answer.copy()
                                adjusted_answer['start'] = answer_start - start
                                adjusted_clause_answers.append(adjusted_answer)
                        
                        adjusted_answers[clause_type] = adjusted_clause_answers
                        
                        # Update clause labels based on whether answers exist in window
                        new_row['clause_labels'][clause_type] = len(adjusted_clause_answers) > 0
                    
                    new_row['clause_answers'] = adjusted_answers
                    new_row['num_positive_clauses'] = sum(new_row['clause_labels'].values())
                    
                    processed_rows.append(new_row)
                    
                    if end >= len(context):
                        break
                    
                    start += window_size - overlap
                    window_count += 1
        
        return pd.DataFrame(processed_rows)
    
    def create_stratified_splits(self, df: pd.DataFrame, test_size: float = 0.2, 
                               val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified splits considering class imbalance
        """
        # Create multi-label matrix for stratification
        clause_matrix = []
        for _, row in df.iterrows():
            clause_vector = [row['clause_labels'].get(clause, False) 
                           for clause in self.clause_types]
            clause_matrix.append(clause_vector)
        
        clause_matrix = np.array(clause_matrix)
        
        # For multi-label stratification, we'll use the most common clause combinations
        clause_combinations = [tuple(row) for row in clause_matrix]
        combination_counts = Counter(clause_combinations)
        
        # Group rare combinations together
        min_samples = 5
        stratify_labels = []
        for combo in clause_combinations:
            if combination_counts[combo] >= min_samples:
                stratify_labels.append(str(combo))
            else:
                stratify_labels.append('rare_combination')
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, 
            stratify=stratify_labels, 
            random_state=42
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_val_labels = [stratify_labels[i] for i in train_val_df.index]
        
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio,
            stratify=train_val_labels,
            random_state=42
        )
        
        return train_df, val_df, test_df
    
    def balance_clause_types(self, df: pd.DataFrame, min_samples: int = 50) -> pd.DataFrame:
        """
        Apply techniques to handle class imbalance
        Based on your finding of severe imbalance
        """
        # Identify rare clause types (< 10% presence from your analysis)
        clause_stats = {}
        for clause in self.clause_types:
            positive_count = sum(1 for _, row in df.iterrows() 
                               if row['clause_labels'].get(clause, False))
            clause_stats[clause] = {
                'count': positive_count,
                'ratio': positive_count / len(df)
            }
        
        # For very rare clauses, consider upsampling or special handling
        rare_clauses = [clause for clause, stats in clause_stats.items() 
                       if stats['count'] < min_samples]
        
        print(f"Identified {len(rare_clauses)} rare clause types requiring special handling:")
        for clause in rare_clauses[:5]:  # Show first 5
            clean_name = self.clean_clause_names.get(clause, clause[:30])
            print(f"  - {clean_name}: {clause_stats[clause]['count']} samples")
        
        # You can implement upsampling here if needed
        return df
    
    def create_model_ready_datasets(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create datasets ready for different model types
        """
        datasets = {}
        
        # 1. Multi-label classification dataset with clean clause names
        ml_rows = []
        for _, row in df.iterrows():
            # Convert clause labels to clean names
            clean_labels = []
            for clause, present in row['clause_labels'].items():
                if present:
                    clean_name = self.clean_clause_names.get(clause, clause)
                    clean_labels.append(clean_name)
            
            ml_rows.append({
                'text': row['context'],
                'labels': clean_labels,
                'document_title': row['document_title'],
                'context_length': row['context_length']
            })
        datasets['multi_label'] = pd.DataFrame(ml_rows)
        
        # 2. Create a mapping for clause types to shorter file-safe names
        clause_name_mapping = {}
        for i, clause_type in enumerate(self.clause_types):
            clean_name = self.clean_clause_names[clause_type]
            
            # Create file-safe version for dataset names
            file_safe_name = clean_name.replace(' ', '_').replace('-', '_').lower()
            file_safe_name = re.sub(r'[^a-zA-Z0-9_]', '', file_safe_name)  # Remove special chars
            
            # Ensure uniqueness and reasonable length
            file_safe_name = file_safe_name[:30]  # Max 30 chars
            if file_safe_name in clause_name_mapping.values():
                file_safe_name = f"{file_safe_name}_{i:02d}"
            
            clause_name_mapping[clause_type] = file_safe_name
        
        # Save the mapping for reference
        self.clause_name_mapping = clause_name_mapping
        
        # 3. Binary classification datasets (one per clause type) with clean names
        for clause_type in self.clause_types:
            binary_rows = []
            clean_name = self.clean_clause_names[clause_type]
            
            for _, row in df.iterrows():
                binary_rows.append({
                    'text': row['context'],
                    'label': int(row['clause_labels'].get(clause_type, False)),
                    'clause_type': clean_name,  # Use clean name instead of verbose question
                    'original_clause_type': clause_type,  # Keep original for reference
                    'document_title': row['document_title']
                })
            
            file_safe_name = clause_name_mapping[clause_type]
            datasets[f'binary_{file_safe_name}'] = pd.DataFrame(binary_rows)
        
        # 4. Question-answering dataset with clean clause names
        qa_rows = []
        for _, row in df.iterrows():
            for clause_type, answers in row['clause_answers'].items():
                if answers:  # Only include examples with answers
                    clean_name = self.clean_clause_names[clause_type]
                    for answer in answers:
                        qa_rows.append({
                            'context': row['context'],
                            'question': f"Find {clean_name} clause",  # Use clean name in question
                            'original_question': clause_type,  # Keep original for reference
                            'answer_text': answer['text'],
                            'answer_start': answer['start'],
                            'clause_type': clean_name,
                            'document_title': row['document_title']
                        })
        datasets['question_answering'] = pd.DataFrame(qa_rows)
        
        return datasets
    
    def process_complete_pipeline(self) -> None:
        """
        Run the complete preprocessing pipeline
        """
        print("Starting CUAD data preprocessing pipeline...")
        
        # 1. Load data
        print("1. Loading CUAD JSON data...")
        cuad_data = self.load_cuad_data()
        
        # 2. Parse to DataFrame
        print("2. Parsing JSON to DataFrame...")
        df = self.parse_cuad_to_dataframe(cuad_data)
        
        # 3. Extract clause types and create clean name mapping
        all_clause_types = set()
        for _, row in df.iterrows():
            all_clause_types.update(row['clause_labels'].keys())
        self.clause_types = sorted(list(all_clause_types))
        
        # Create clean clause names mapping
        print("3. Creating clean clause name mappings...")
        for clause_type in self.clause_types:
            clean_name = extract_clause_name(clause_type)
            self.clean_clause_names[clause_type] = clean_name
        
        print(f"   Found {len(self.clause_types)} clause types")
        print(f"   Total contexts: {len(df)}")
        print(f"   Sample clean names:")
        for i, (verbose, clean) in enumerate(list(self.clean_clause_names.items())[:5]):
            print(f"     '{clean}' (from: {verbose[:50]}...)")
        
        # 4. Clean text
        print("4. Cleaning text data...")
        df['context'] = df['context'].apply(self.clean_text)
        
        # 5. Handle long contexts
        print("5. Handling long contexts with sliding window...")
        df = self.handle_long_contexts(df)
        print(f"   After windowing: {len(df)} contexts")
        
        # 6. Balance clause types (optional)
        print("6. Analyzing class imbalance...")
        df = self.balance_clause_types(df)
        
        # 7. Create stratified splits
        print("7. Creating stratified train/val/test splits...")
        train_df, val_df, test_df = self.create_stratified_splits(df)
        
        print(f"   Train: {len(train_df)} samples")
        print(f"   Val: {len(val_df)} samples") 
        print(f"   Test: {len(test_df)} samples")
        
        # 8. Create model-ready datasets
        print("8. Creating model-ready datasets with clean clause names...")
        train_datasets = self.create_model_ready_datasets(train_df)
        val_datasets = self.create_model_ready_datasets(val_df)
        test_datasets = self.create_model_ready_datasets(test_df)
        
        # 9. Save processed data
        print("9. Saving processed datasets...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save raw splits
        train_df.to_pickle(f"{self.output_dir}/train_raw.pkl")
        val_df.to_pickle(f"{self.output_dir}/val_raw.pkl") 
        test_df.to_pickle(f"{self.output_dir}/test_raw.pkl")
        
        # Save model-ready datasets
        for dataset_name, dataset in train_datasets.items():
            dataset.to_csv(f"{self.output_dir}/train_{dataset_name}.csv", index=False)
        
        for dataset_name, dataset in val_datasets.items():
            dataset.to_csv(f"{self.output_dir}/val_{dataset_name}.csv", index=False)
            
        for dataset_name, dataset in test_datasets.items():
            dataset.to_csv(f"{self.output_dir}/test_{dataset_name}.csv", index=False)
        
        # Save metadata with clean clause names
        metadata = {
            'clause_types': self.clause_types,
            'clean_clause_names': self.clean_clause_names,  # Add clean names mapping
            'clause_name_mapping': getattr(self, 'clause_name_mapping', {}),
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'dataset_types': list(train_datasets.keys())
        }

        with open(f"{self.output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Preprocessing pipeline completed!")
        print(f"Outputs saved to: {self.output_dir}")
        print(f"Available datasets: {list(train_datasets.keys())}")
        print(f"Clean clause names saved in metadata for easier analysis!")


def main():
    """Main function to run preprocessing"""
    # Get the script directory and build absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'CUAD_v1', 'CUAD_v1.json')
    output_dir = os.path.join(project_root, 'data', 'processed')
    
    print(f"Looking for CUAD data at: {raw_data_path}")
    print(f"File exists: {os.path.exists(raw_data_path)}")
    
    preprocessor = CUADPreprocessor(
        raw_data_path=raw_data_path,
        output_dir=output_dir
    )
    
    preprocessor.process_complete_pipeline()


if __name__ == "__main__":
    main()