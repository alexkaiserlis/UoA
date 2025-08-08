#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eurovoc Concept Classifier using TF-IDF on Generated Summaries

This system classifies input text to the most relevant Eurovoc concepts
using TF-IDF similarity on our generated concept summaries.

Features:
- TF-IDF vectorization with configurable parameters
- Top-K concept retrieval with confidence scores
- Evaluation metrics (Recall@K, ZSR@K)
- Support for unseen label evaluation
- Comprehensive performance analytics
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import re

# TF-IDF και ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# PDF processing imports
try:
    import PyPDF2
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Greek stopwords (μπορούμε να επεκτείνουμε)
GREEK_STOPWORDS = {
    'και', 'ή', 'των', 'της', 'του', 'το', 'τα', 'στο', 'στη', 'στην', 'στα', 
    'για', 'με', 'από', 'προς', 'κατά', 'μετά', 'χωρίς', 'παρά', 'εις', 'ως',
    'είναι', 'ήταν', 'θα', 'να', 'ότι', 'αν', 'όταν', 'όπου', 'όπως', 'αλλά',
    'μια', 'ένα', 'μας', 'σας', 'τους', 'τις', 'τον', 'την', 'όλα', 'όλες',
    'πως', 'πού', 'τι', 'ποια', 'ποιο', 'ποιες', 'ποιους', 'ποιων'
}

ENGLISH_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 
    'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 
    'will', 'with', 'or', 'but', 'not', 'this', 'these', 'they', 'them',
    'have', 'had', 'can', 'could', 'should', 'would', 'may', 'might',
    'all', 'any', 'each', 'every', 'some', 'such', 'no', 'nor', 'too',
    'very', 'just', 'now', 'only', 'also', 'into', 'over', 'after',
    'other', 'which', 'their', 'what', 'there', 'when', 'where', 'who',
    'how', 'up', 'out', 'if', 'about', 'were', 'been', 'being', 'do',
    'does', 'did', 'doing', 'than', 'through', 'during', 'before'
}

ALL_STOPWORDS = GREEK_STOPWORDS.union(ENGLISH_STOPWORDS)

@dataclass
class ClassificationResult:
    """Αποτέλεσμα classification για ένα κείμενο."""
    query_text: str
    predicted_concepts: List[Tuple[str, str, float]]  # (concept_id, title, score)
    true_concepts: Optional[Set[str]] = None
    processing_time: float = 0.0
    query_length: int = 0

@dataclass
class EvaluationMetrics:
    """Μετρικές αξιολόγησης για Top-K evaluation."""
    k: int
    recall_at_k: float
    zsr_at_k: float  # Zero-Shot Recall
    precision_at_k: float = 0.0
    f1_at_k: float = 0.0

class EurovocClassifier:
    """
    Eurovoc Concept Classifier using TF-IDF on concept summaries.
    """
    
    def __init__(self, summaries_dir: str, config: Optional[Dict] = None):
        """
        Αρχικοποίηση classifier.
        
        Args:
            summaries_dir: Κατάλογος με τα summary JSON αρχεία
            config: Παραμετροποίηση TF-IDF και classifier
        """
        self.summaries_dir = os.path.abspath(summaries_dir)
        self.config = config or self._default_config()
        
        # Data containers
        self.concept_summaries = {}  # concept_id -> summary_text
        self.concept_metadata = {}   # concept_id -> full_metadata
        self.vectorizer = None
        self.tfidf_matrix = None
        self.concept_ids = []        # Ordered list of concept IDs
        
        # Performance tracking
        self.classification_stats = {
            'total_queries': 0,
            'total_concepts_loaded': 0,
            'avg_processing_time': 0.0,
            'last_update': None
        }
    
    def _default_config(self) -> Dict:
        """Προεπιλεγμένη παραμετροποίηση συστήματος."""
        return {
            'tfidf_params': {
                'lowercase': True,
                'ngram_range': (1, 1),  # Unigrams to trigrams
                'stop_words': list(ALL_STOPWORDS),
                'max_features': 50000,
                'min_df': 2,            # Ignore terms in less than 2 documents
                'max_df': 0.95,         # Ignore terms in more than 95% of documents
                'sublinear_tf': True,   # Use log-scaled term frequencies
                'norm': 'l2'            # L2 normalization
            },
            'classification_params': {
                'similarity_metric': 'cosine',
                'top_k_default': 10,
                'min_score_threshold': 0.01
            },
            'evaluation_params': {
                'top_k_values': [1, 3, 5, 10, 50],
                'include_zsr': True,    # Zero-Shot Recall
                'unseen_label_ratio': 0.3
            }
        }
    
    def load_summaries(self, verbose: bool = True) -> bool:
        """
        Φόρτωση όλων των concept summaries από τον κατάλογο.
        
        Returns:
            bool: True αν η φόρτωση ήταν επιτυχής
        """
        if verbose:
            print(f"📂 Φόρτωση summaries από: {self.summaries_dir}")
        
        if not os.path.exists(self.summaries_dir):
            print(f"❌ Κατάλογος δεν υπάρχει: {self.summaries_dir}")
            return False
        
        summary_files = [f for f in os.listdir(self.summaries_dir) 
                        if f.endswith('_summary.json')]
        
        if not summary_files:
            print(f"❌ Δεν βρέθηκαν summary αρχεία στο {self.summaries_dir}")
            return False
        
        loaded = 0
        failed = 0
        
        for filename in summary_files:
            try:
                # Extract concept ID από το filename
                # Format: concept_XXXX_summary.json
                concept_id = filename.replace('concept_', '').replace('_summary.json', '')
                
                filepath = os.path.join(self.summaries_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Εξαγωγή summary text
                summary_text = data.get('summary', '')
                if not summary_text:
                    print(f"⚠️  Κενή περίληψη για concept {concept_id}")
                    continue
                
                self.concept_summaries[concept_id] = summary_text
                self.concept_metadata[concept_id] = data
                loaded += 1
                
            except Exception as e:
                print(f"❌ Σφάλμα φόρτωσης {filename}: {e}")
                failed += 1
                continue
        
        if verbose:
            print(f"✅ Φορτώθηκαν {loaded:,} concept summaries")
            if failed > 0:
                print(f"⚠️  {failed} αποτυχίες φόρτωσης")
        
        self.classification_stats['total_concepts_loaded'] = loaded
        self.classification_stats['last_update'] = datetime.now().isoformat()
        
        return loaded > 0
    
    def build_tfidf_index(self, verbose: bool = True) -> bool:
        """
        Δημιουργία TF-IDF index από τα summaries.
        
        Returns:
            bool: True αν η δημιουργία ήταν επιτυχής
        """
        if not self.concept_summaries:
            print("❌ Δεν υπάρχουν summaries. Καλέστε πρώτα load_summaries()")
            return False
        
        if verbose:
            print(f"🔧 Δημιουργία TF-IDF index με {len(self.concept_summaries):,} concepts...")
        
        # Προετοιμασία δεδομένων
        self.concept_ids = list(self.concept_summaries.keys())
        summary_texts = [self.concept_summaries[cid] for cid in self.concept_ids]
        
        # TF-IDF vectorization
        try:
            self.vectorizer = TfidfVectorizer(**self.config['tfidf_params'])
            self.tfidf_matrix = self.vectorizer.fit_transform(summary_texts)
            
            if verbose:
                print(f"✅ TF-IDF index δημιουργήθηκε:")
                print(f"   • Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
                print(f"   • Matrix shape: {self.tfidf_matrix.shape}")
                print(f"   • Sparsity: {(1 - self.tfidf_matrix.nnz / np.prod(self.tfidf_matrix.shape)):.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Σφάλμα δημιουργίας TF-IDF index: {e}")
            return False
    
    def classify_text(self, query_text: str, top_k: int = 10, 
                     return_scores: bool = True) -> ClassificationResult:
        """
        Classification κειμένου σε Eurovoc concepts.
        
        Args:
            query_text: Το κείμενο προς classification
            top_k: Αριθμός top concepts να επιστραφούν
            return_scores: Αν να συμπεριληφθούν τα scores
            
        Returns:
            ClassificationResult: Αποτελέσματα classification
        """
        start_time = datetime.now()
        
        if not self.vectorizer or self.tfidf_matrix is None:
            raise ValueError("TF-IDF index δεν έχει δημιουργηθεί. Καλέστε build_tfidf_index()")
        
        # Vectorization του query
        query_vector = self.vectorizer.transform([query_text])
        
        # Υπολογισμός similarity με όλα τα concepts
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Top-K concepts
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Δημιουργία results
        predicted_concepts = []
        min_threshold = self.config['classification_params']['min_score_threshold']
        
        for idx in top_indices:
            concept_id = self.concept_ids[idx]
            score = similarities[idx]
            
            if score < min_threshold:
                break
                
            # Παίρνουμε τον τίτλο από metadata
            title = self.concept_metadata[concept_id].get('concept_data', {}).get('title', 'Unknown')
            predicted_concepts.append((concept_id, title, score))
        
        # Timing
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = ClassificationResult(
            query_text=query_text,
            predicted_concepts=predicted_concepts,
            processing_time=processing_time,
            query_length=len(query_text.split())
        )
        
        # Update stats
        self.classification_stats['total_queries'] += 1
        self.classification_stats['avg_processing_time'] = (
            (self.classification_stats['avg_processing_time'] * (self.classification_stats['total_queries'] - 1) + 
             processing_time) / self.classification_stats['total_queries']
        )
        
        return result
    
    def evaluate_on_dataset(self, test_data: List[Dict], 
                           top_k_values: Optional[List[int]] = None,
                           verbose: bool = True) -> Dict[int, EvaluationMetrics]:
        """
        Αξιολόγηση συστήματος σε test dataset.
        
        Args:
            test_data: List of {'text': str, 'concepts': List[str]}
            top_k_values: List of K values for evaluation
            verbose: Print progress
            
        Returns:
            Dict mapping K -> EvaluationMetrics
        """
        if top_k_values is None:
            top_k_values = self.config['evaluation_params']['top_k_values']
        
        if verbose:
            print(f"📊 Αξιολόγηση σε {len(test_data):,} test samples...")
        
        results = {}
        max_k = max(top_k_values)
        
        # Classification για όλα τα test samples
        all_predictions = []
        all_true_labels = []
        
        for i, sample in enumerate(test_data):
            if verbose and (i + 1) % 100 == 0:
                print(f"   Progress: {i+1:,}/{len(test_data):,}")
            
            text = sample['text']
            true_concepts = set(sample['concepts'])
            
            # Classification
            result = self.classify_text(text, top_k=max_k)
            predicted_concepts = [pred[0] for pred in result.predicted_concepts]
            
            all_predictions.append(predicted_concepts)
            all_true_labels.append(true_concepts)
        
        # Υπολογισμός μετρικών για κάθε K
        for k in top_k_values:
            recall_scores = []
            zsr_scores = []  # Zero-Shot Recall
            
            for predicted, true_set in zip(all_predictions, all_true_labels):
                predicted_k = predicted[:k]
                predicted_set = set(predicted_k)
                
                # Recall@K
                if len(true_set) > 0:
                    recall = len(predicted_set.intersection(true_set)) / len(true_set)
                    recall_scores.append(recall)
                
                # ZSR@K (Zero-Shot Recall)
                if len(predicted_set) > 0:
                    zsr = len(predicted_set.intersection(true_set)) / len(predicted_set)
                    zsr_scores.append(zsr)
            
            # Μέσοι όροι
            avg_recall = np.mean(recall_scores) if recall_scores else 0.0
            avg_zsr = np.mean(zsr_scores) if zsr_scores else 0.0
            
            results[k] = EvaluationMetrics(
                k=k,
                recall_at_k=avg_recall,
                zsr_at_k=avg_zsr
            )
        
        if verbose:
            print("✅ Αξιολόγηση ολοκληρώθηκε!")
            self._print_evaluation_results(results)
        
        return results
    
    def _print_evaluation_results(self, results: Dict[int, EvaluationMetrics]):
        """Εκτύπωση αποτελεσμάτων αξιολόγησης."""
        print(f"\n📈 EVALUATION RESULTS:")
        print("=" * 50)
        
        for k in sorted(results.keys()):
            metrics = results[k]
            print(f"\nTop-{k} Evaluation")
            if k == 1:
                print("        with unseen labels")
            print(f"   🔹 Recall@{k}: {metrics.recall_at_k:.4f}")
            print(f"   🔹 ZSR@{k}:  {metrics.zsr_at_k:.4f}")
    
    def save_evaluation_report(self, results: Dict[int, EvaluationMetrics], 
                              output_file: str, test_info: Optional[Dict] = None):
        """Αποθήκευση αναλυτικής αναφοράς αξιολόγησης."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'classifier_config': self.config,
            'classification_stats': self.classification_stats,
            'test_info': test_info or {},
            'evaluation_results': {
                str(k): {
                    'k': metrics.k,
                    'recall_at_k': metrics.recall_at_k,
                    'zsr_at_k': metrics.zsr_at_k,
                    'precision_at_k': metrics.precision_at_k,
                    'f1_at_k': metrics.f1_at_k
                }
                for k, metrics in results.items()
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📄 Evaluation report αποθηκεύτηκε: {output_file}")
    
    def get_system_info(self) -> Dict:
        """Πληροφορίες συστήματος και στατιστικά."""
        vocab_size = len(self.vectorizer.vocabulary_) if self.vectorizer else 0
        matrix_shape = self.tfidf_matrix.shape if self.tfidf_matrix is not None else (0, 0)
        
        return {
            'loaded_concepts': len(self.concept_summaries),
            'vocabulary_size': vocab_size,
            'tfidf_matrix_shape': matrix_shape,
            'classification_stats': self.classification_stats,
            'config': self.config
        }

def extract_pdf_text(pdf_path: str, max_chars: int = 50000) -> str:
    """
    Εξαγωγή κειμένου από PDF αρχείο.
    
    Args:
        pdf_path: Διαδρομή προς το PDF αρχείο
        max_chars: Μέγιστος αριθμός χαρακτήρων να εξαχθούν
        
    Returns:
        str: Εξαγμένο κείμενο από το PDF
    """
    if not os.path.exists(pdf_path):
        return ""
    
    text = ""
    
    # Προσπάθεια με PyMuPDF (fitz) πρώτα - καλύτερο για complex PDFs
    if PDF_AVAILABLE:
        try:
            import fitz
            doc = fitz.open(pdf_path)
            for page_num in range(min(10, len(doc))):  # Πρώτες 10 σελίδες
                page = doc.load_page(page_num)
                text += page.get_text()
                if len(text) > max_chars:
                    break
            doc.close()
            
            if text.strip():
                return text[:max_chars]
                
        except Exception as e:
            print(f"   ⚠️  PyMuPDF failed για {os.path.basename(pdf_path)}: {e}")
    
    # Fallback σε PyPDF2
    if PDF_AVAILABLE:
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(min(10, len(pdf_reader.pages))):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                    if len(text) > max_chars:
                        break
                        
            if text.strip():
                return text[:max_chars]
                
        except Exception as e:
            print(f"   ⚠️  PyPDF2 failed για {os.path.basename(pdf_path)}: {e}")
    
    # Τελευταίο fallback - manual text extraction
    try:
        # Αν δεν έχουμε PDF libraries, προσπαθούμε να διαβάσουμε ως text
        with open(pdf_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read(max_chars)
        return text
    except:
        return ""

def load_test_pdfs(test_folder: str, max_files: int = 5) -> List[Dict]:
    """
    Φόρτωση PDF αρχείων από test folder για classification testing.
    
    Args:
        test_folder: Φάκελος με PDF αρχεία
        max_files: Μέγιστος αριθμός αρχείων να φορτωθούν
        
    Returns:
        List of test samples με format {'text': str, 'filename': str, 'concepts': List[str]}
    """
    if not PDF_AVAILABLE:
        print("⚠️  PDF libraries δεν είναι διαθέσιμες!")
        print("   Εγκαταστήστε με: pip install PyPDF2 PyMuPDF")
        return []
    
    print(f"📂 Φόρτωση PDF αρχείων από: {test_folder}")
    
    if not os.path.exists(test_folder):
        print(f"❌ Φάκελος δεν υπάρχει: {test_folder}")
        return []
    
    # Βρίσκουμε όλα τα PDF αρχεία
    pdf_files = [f for f in os.listdir(test_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ Δεν βρέθηκαν PDF αρχεία στο {test_folder}")
        return []
    
    # Περιορίζουμε τον αριθμό αρχείων
    pdf_files = pdf_files[:max_files]
    
    print(f"📄 Βρέθηκαν {len(pdf_files)} PDF αρχεία")
    
    test_samples = []
    
    for i, filename in enumerate(pdf_files, 1):
        print(f"   {i}/{len(pdf_files)} Επεξεργασία: {filename}...")
        
        pdf_path = os.path.join(test_folder, filename)
        
        # Εξαγωγή κειμένου
        text = extract_pdf_text(pdf_path, max_chars=30000)  # ~30K chars για καλή απόδοση
        
        if not text.strip():
            print(f"   ⚠️  Δεν εξήχθη κείμενο από {filename}")
            continue
        
        # Καθαρισμός κειμένου
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        if len(text) < 100:  # Πολύ μικρό κείμενο
            print(f"   ⚠️  Πολύ μικρό κείμενο για {filename}")
            continue
        
        # Δημιουργία test sample
        sample = {
            'text': text,
            'filename': filename,
            'concepts': [],  # Θα τα συμπληρώσετε χειροκίνητα ή από metadata
            'text_length': len(text.split()),
            'char_length': len(text)
        }
        
        test_samples.append(sample)
        print(f"   ✅ Φορτώθηκε: {len(text.split())} λέξεις, {len(text)} χαρακτήρες")
    
    print(f"✅ Συνολικά φορτώθηκαν {len(test_samples)} PDF αρχεία")
    
    return test_samples

def main():
    """Demo και testing του classifier με PDF αρχεία."""
    print("🎯 EUROVOC CONCEPT CLASSIFIER - PDF TESTING")
    print("=" * 60)
    
    # Παραμετροποίηση paths
    current_dir = os.path.dirname(__file__)
    summaries_dir = os.path.join(current_dir, 'summaries')
    
    # Φάκελος με test PDF αρχεία (relative στο IR directory)
    parent_dir = os.path.dirname(current_dir)  # FINAL_VOCABULARY_PACKAGE
    test_pdfs_dir = os.path.join(parent_dir, 'TEST_LEGISLATIONS')
    
    print(f"📁 Summaries directory: {summaries_dir}")
    print(f"📁 Test PDFs directory: {test_pdfs_dir}")
    
    # Δημιουργία classifier
    classifier = EurovocClassifier(summaries_dir)
    
    # Φόρτωση δεδομένων
    print("\n🔧 Αρχικοποίηση classifier...")
    if not classifier.load_summaries():
        print("❌ Αποτυχία φόρτωσης summaries")
        print("   Βεβαιωθείτε ότι έχετε τρέξει batch summary generation πρώτα")
        return
    
    # Δημιουργία TF-IDF index
    if not classifier.build_tfidf_index():
        print("❌ Αποτυχία δημιουργίας TF-IDF index")
        return
    
    # Έλεγχος αν υπάρχουν PDF αρχεία
    if not os.path.exists(test_pdfs_dir):
        print(f"\n⚠️  TEST_LEGISLATIONS φάκελος δεν υπάρχει: {test_pdfs_dir}")
        print("   Χρησιμοποιώ default test texts...")
        
        # Fallback σε default test texts
        test_texts = [
            "European Union trade policy and international commerce regulations",
            "Environmental protection and climate change legislation",
            "Healthcare systems and medical research funding",
            "Agricultural subsidies and farming regulations",
            "Digital privacy and data protection laws"
        ]
        
        print(f"\n🧪 ΔΟΚΙΜΗ CLASSIFICATION (Default texts):")
        print("-" * 40)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n{i}. Query: {text[:50]}...")
            
            result = classifier.classify_text(text, top_k=5)
            
            print(f"   ⏱️  Processing time: {result.processing_time:.3f}s")
            print(f"   🎯 Top-5 concepts:")
            
            for j, (concept_id, title, score) in enumerate(result.predicted_concepts, 1):
                print(f"      {j}. {title} (ID: {concept_id}) - Score: {score:.4f}")
    
    else:
        # Φόρτωση PDF αρχείων
        print(f"\n📚 Φόρτωση PDF αρχείων από TEST_LEGISLATIONS...")
        test_samples = load_test_pdfs(test_pdfs_dir, max_files=3)  # Περιορίζουμε σε 3 για demo
        
        if not test_samples:
            print("❌ Δεν φορτώθηκαν PDF αρχεία")
            return
        
        print(f"\n🧪 ΔΟΚΙΜΗ CLASSIFICATION (PDF Legislations):")
        print("-" * 50)
        
        for i, sample in enumerate(test_samples, 1):
            filename = sample['filename']
            text = sample['text']
            word_count = sample['text_length']
            
            print(f"\n{i}. PDF: {filename}")
            print(f"   📊 Μέγεθος: {word_count:,} λέξεις")
            
            # Παίρνουμε τα πρώτα 200 χαρακτήρες για preview
            preview = text[:200].replace('\n', ' ').replace('\r', ' ')
            preview = re.sub(r'\s+', ' ', preview).strip()
            print(f"   📝 Preview: {preview}...")
            
            # Classification
            start_time = datetime.now()
            result = classifier.classify_text(text, top_k=10)
            
            print(f"   ⏱️  Processing time: {result.processing_time:.3f}s")
            print(f"   🎯 Top-10 concepts:")
            
            for j, (concept_id, title, score) in enumerate(result.predicted_concepts, 1):
                # Color coding για scores
                if score > 0.3:
                    color = "🟢"
                elif score > 0.1:
                    color = "🟡"
                else:
                    color = "🔴"
                
                print(f"      {j:2d}. {color} {title}")
                print(f"          ID: {concept_id} | Score: {score:.4f}")
            
            print(f"\n   💡 Οι υψηλότερες ομοιότητες για {filename}:")
            top_3 = result.predicted_concepts[:3]
            for rank, (cid, title, score) in enumerate(top_3, 1):
                print(f"      #{rank}: {title} ({score:.3f})")
    
    # System info
    info = classifier.get_system_info()
    print(f"\n📊 SYSTEM INFO:")
    print("-" * 30)
    print(f"   • Loaded concepts: {info['loaded_concepts']:,}")
    print(f"   • Vocabulary size: {info['vocabulary_size']:,}")
    print(f"   • TF-IDF matrix: {info['tfidf_matrix_shape']}")
    print(f"   • Total queries processed: {info['classification_stats']['total_queries']}")
    
    if info['classification_stats']['total_queries'] > 0:
        avg_time = info['classification_stats']['avg_processing_time']
        print(f"   • Average processing time: {avg_time:.3f}s")
    
    print(f"\n✅ Demo ολοκληρώθηκε!")
    print(f"📋 Για πιο λεπτομερή evaluation, τρέξτε: python evaluate_classifier.py")
    print(f"🎮 Για interactive testing, τρέξτε: python demo_classifier.py")

if __name__ == "__main__":
    main()
