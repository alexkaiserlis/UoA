"""
Advanced Evaluation Metrics for Greek Legal NER

This module provides comprehensive evaluation metrics and analysis functions
for Named Entity Recognition tasks, with special focus on Greek Legal NER.
It includes entity-level metrics, tag-level analysis, BIO validation,
and detailed per-class performance breakdowns.

🔍 EVALUATION STRATEGY:
======================

1. **Multi-Level Analysis**:
   - Entity-level metrics (SeqEval standard)
   - Tag-level metrics (training-relevant)
   - Sequence-level validation (BIO consistency)

2. **Comprehensive Coverage**:
   - Per-entity type breakdown
   - Rare class analysis
   - Boundary error analysis
   - Transition pattern analysis

3. **Research-Grade Metrics**:
   - Standard NER metrics (P/R/F1)
   - Class-balanced metrics
   - Error categorization
   - Performance visualization data

📊 KEY METRICS EXPLAINED:
========================

**Entity-Level F1**: Πραγματική NER performance (complete entity match)
**Macro F1**: Unweighted average across entity types (rare class friendly)
**Micro F1**: Frequency-weighted average (common class dominated)
**Tag Accuracy**: Individual token classification accuracy
**Boundary Accuracy**: Correct entity start/end detection
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
from dataclasses import dataclass
import evaluate
import json


# =============================================================================
# JSON SERIALIZATION HELPER
# =============================================================================

def clean_for_json_serialization(obj):
    """
    Recursively clean data structure to make it JSON serializable.
    Converts NumPy types to native Python types.
    
    Args:
        obj: Object to clean
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {key: clean_for_json_serialization(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json_serialization(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    else:
        return obj


# =============================================================================
# CORE METRIC COMPUTATION CLASSES
# =============================================================================

@dataclass
class EntityMetrics:
    """
    Container for entity-level metrics.
    
    Stores comprehensive metrics for a single entity type including
    precision, recall, F1, support, and detailed tag-level breakdowns.
    """
    entity_type: str
    
    # Entity-level metrics (SeqEval)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0  # Number of true entities
    
    # Tag-level breakdown
    b_tag_metrics: Dict[str, Union[int, float]] = None
    i_tag_metrics: Dict[str, Union[int, float]] = None
    
    # Error analysis
    boundary_errors: int = 0
    type_confusion_errors: int = 0
    missing_entities: int = 0
    spurious_entities: int = 0
    
    def __post_init__(self):
        if self.b_tag_metrics is None:
            self.b_tag_metrics = {
                'true': 0, 'pred': 0, 'correct': 0, 'accuracy': 0.0
            }
        if self.i_tag_metrics is None:
            self.i_tag_metrics = {
                'true': 0, 'pred': 0, 'correct': 0, 'accuracy': 0.0
            }
    
    def get_combined_f1(self) -> float:
        """
        Compute combined F1 score for B- and I- tags of this entity type.
        
        This is different from entity-level F1 as it treats each tag
        independently rather than requiring complete entity matches.
        """
        total_true = self.b_tag_metrics['true'] + self.i_tag_metrics['true']
        total_pred = self.b_tag_metrics['pred'] + self.i_tag_metrics['pred']
        total_correct = self.b_tag_metrics['correct'] + self.i_tag_metrics['correct']
        
        if total_pred == 0:
            precision = 0.0
        else:
            precision = total_correct / total_pred
        
        if total_true == 0:
            recall = 0.0
        else:
            recall = total_correct / total_true
        
        if precision + recall == 0:
            return 0.0
        else:
            return 2 * precision * recall / (precision + recall)


@dataclass 
class EvaluationResults:
    """
    Complete evaluation results container.
    
    Stores all computed metrics in a structured format for easy access
    and serialization.
    """
    
    # Overall performance
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0
    overall_accuracy: float = 0.0
    
    # Aggregated metrics
    macro_f1: float = 0.0  # Unweighted average across entity types
    micro_f1: float = 0.0  # Frequency-weighted average
    tag_level_accuracy: float = 0.0
    
    # Per-entity metrics
    entity_metrics: Dict[str, EntityMetrics] = None
    
    # Sequence-level analysis
    total_sequences: int = 0
    invalid_sequences: int = 0
    orphaned_i_tags: int = 0
    sequence_coherence_score: float = 0.0
    
    # Error analysis
    total_entities_true: int = 0
    total_entities_pred: int = 0
    boundary_errors: int = 0
    classification_errors: int = 0
    
    # CRF-specific metrics (if applicable)
    crf_metrics: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.entity_metrics is None:
            self.entity_metrics = {}
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get a summary dictionary suitable for logging/saving."""
        summary = {
            'overall_f1': self.overall_f1,
            'overall_precision': self.overall_precision,
            'overall_recall': self.overall_recall,
            'macro_f1': self.macro_f1,
            'micro_f1': self.micro_f1,
            'tag_accuracy': self.tag_level_accuracy,
            'total_entities_true': self.total_entities_true,
            'total_entities_pred': self.total_entities_pred,
            'invalid_sequences': self.invalid_sequences,
            'orphaned_i_tags': self.orphaned_i_tags
        }
        
        # Add per-entity F1 scores
        for entity_type, metrics in self.entity_metrics.items():
            summary[f'{entity_type}_f1'] = metrics.f1
            summary[f'{entity_type}_precision'] = metrics.precision
            summary[f'{entity_type}_recall'] = metrics.recall
            summary[f'{entity_type}_support'] = metrics.support
            
            # Add detailed tag-level metrics for each entity
            if metrics.b_tag_metrics:
                summary[f'B-{entity_type}_true_count'] = metrics.b_tag_metrics['true']
                summary[f'B-{entity_type}_pred_count'] = metrics.b_tag_metrics['pred']
                summary[f'B-{entity_type}_correct_count'] = metrics.b_tag_metrics['correct']
                summary[f'B-{entity_type}_accuracy'] = metrics.b_tag_metrics['accuracy']
            
            if metrics.i_tag_metrics:
                summary[f'I-{entity_type}_true_count'] = metrics.i_tag_metrics['true']
                summary[f'I-{entity_type}_pred_count'] = metrics.i_tag_metrics['pred']
                summary[f'I-{entity_type}_correct_count'] = metrics.i_tag_metrics['correct']
                summary[f'I-{entity_type}_accuracy'] = metrics.i_tag_metrics['accuracy']
        
        # Add CRF metrics if available
        if self.crf_metrics:
            summary.update(self.crf_metrics)
        
        return summary


# =============================================================================
# CORE EVALUATION FUNCTIONS
# =============================================================================

class NERMetricsCalculator:
    """
    Advanced NER metrics calculator with comprehensive analysis capabilities.
    
    🔍 CALCULATION METHODOLOGY:
    ==========================
    
    **Entity-Level Metrics (Primary)**:
    - Uses SeqEval library for standard NER evaluation
    - Requires exact entity boundary and type match
    - Industry standard for NER evaluation
    
    **Tag-Level Metrics (Supplementary)**:
    - Individual token classification accuracy
    - Useful for training monitoring
    - More granular error analysis
    
    **Sequence-Level Validation**:
    - BIO tagging consistency checks
    - Orphaned I- tag detection
    - Invalid transition identification
    """
    
    def __init__(self, label_list: List[str], use_crf: bool = False):
        """
        Initialize the metrics calculator.
        
        Args:
            label_list: List of all possible labels
            use_crf: Whether CRF is being used (affects some metrics)
        """
        self.label_list = label_list
        self.use_crf = use_crf
        self.id2label = {i: label for i, label in enumerate(label_list)}
        self.label2id = {label: i for i, label in enumerate(label_list)}
        
        # Initialize SeqEval metric
        self.seqeval_metric = evaluate.load("seqeval")
        
        # Entity type extraction
        self.entity_types = self._extract_entity_types(label_list)
    
    def _extract_entity_types(self, label_list: List[str]) -> List[str]:
        """Extract unique entity types from label list."""
        entity_types = set()
        for label in label_list:
            if label.startswith('B-') or label.startswith('I-'):
                entity_type = label[2:]
                entity_types.add(entity_type)
        return sorted(list(entity_types))
    
    def compute_comprehensive_metrics(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray,
        model: Optional[Any] = None,
        verbose: bool = True
    ) -> EvaluationResults:
        """
        Compute comprehensive evaluation metrics.
        
        🔍 COMPUTATION PIPELINE:
        =======================
        1. Convert predictions/labels to string format
        2. Compute SeqEval entity-level metrics  
        3. Compute tag-level metrics and accuracy
        4. Validate BIO sequence consistency
        5. Analyze entity boundaries and errors
        6. Compute CRF-specific metrics (if applicable)
        7. Aggregate all results
        
        Args:
            predictions: Model predictions [batch_size, seq_len]
            labels: True labels [batch_size, seq_len]
            model: Model instance (for CRF analysis)
            verbose: Whether to print detailed analysis
        
        Returns:
            EvaluationResults: Comprehensive evaluation results
        """
        
        # Ensure predictions are in the right format
        if len(predictions.shape) == 3:  # [batch_size, seq_len, num_labels]
            predictions = np.argmax(predictions, axis=2)
        
        # Convert to string labels
        true_labels_str, pred_labels_str = self._convert_to_string_labels(
            predictions, labels
        )
        
        if verbose:
            print(f"\n📊 COMPREHENSIVE METRICS COMPUTATION:")
            print("="*60)
            print(f"   Total sequences: {len(true_labels_str):,}")
            print(f"   Entity types: {len(self.entity_types)}")
        
        # 1. Compute SeqEval entity-level metrics
        seqeval_results = self._compute_seqeval_metrics(
            pred_labels_str, true_labels_str, verbose
        )
        
        # 2. Compute tag-level metrics
        tag_metrics = self._compute_tag_level_metrics(
            predictions, labels, verbose
        )
        
        # 3. Validate BIO consistency
        bio_validation = self._validate_bio_consistency(
            pred_labels_str, verbose
        )
        
        # 4. Analyze entity boundaries
        boundary_analysis = self._analyze_entity_boundaries(
            pred_labels_str, true_labels_str, verbose
        )
        
        # 5. Compute CRF-specific metrics
        crf_metrics = None
        if self.use_crf and model is not None:
            crf_metrics = self._compute_crf_metrics(
                model, predictions, labels, verbose
            )
        
        # 6. Aggregate results
        results = self._aggregate_results(
            seqeval_results, tag_metrics, bio_validation, 
            boundary_analysis, crf_metrics
        )
        
        if verbose:
            self._print_results_summary(results)
        
        return results
    
    def _convert_to_string_labels(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Convert numerical predictions and labels to string format.
        
        Handles -100 labels (padding/ignored tokens) by filtering them out.
        """
        true_labels_str = []
        pred_labels_str = []
        
        for pred_seq, true_seq in zip(predictions, labels):
            # Filter out -100 (ignored) tokens
            valid_indices = true_seq != -100
            
            valid_true = true_seq[valid_indices]
            valid_pred = pred_seq[valid_indices]
            
            # Convert to string labels
            true_seq_str = [self.id2label[label_id] for label_id in valid_true]
            pred_seq_str = [self.id2label[label_id] for label_id in valid_pred]
            
            true_labels_str.append(true_seq_str)
            pred_labels_str.append(pred_seq_str)
        
        return true_labels_str, pred_labels_str
    
    def _compute_seqeval_metrics(
        self, 
        predictions: List[List[str]], 
        references: List[List[str]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute SeqEval entity-level metrics.
        
        🎯 SeqEval METHODOLOGY:
        ======================
        - Extracts complete entities from B-I-O sequences
        - Requires exact boundary and type match
        - Returns per-entity-type and overall metrics
        - Standard evaluation for NER research
        """
        
        seqeval_results = self.seqeval_metric.compute(
            predictions=predictions, 
            references=references, 
            zero_division=0
        )
        
        if verbose:
            print(f"\n🎯 SEQEVAL ENTITY-LEVEL METRICS:")
            print("-" * 40)
            print(f"   Overall F1: {seqeval_results['overall_f1']:.4f}")
            print(f"   Overall Precision: {seqeval_results['overall_precision']:.4f}")
            print(f"   Overall Recall: {seqeval_results['overall_recall']:.4f}")
            
            print(f"\n   Per-Entity Performance:")
            for entity_type in sorted(self.entity_types):
                if entity_type in seqeval_results:
                    metrics = seqeval_results[entity_type]
                    f1 = metrics.get('f1', 0)
                    support = metrics.get('number', 0)
                    status = "🔴" if f1 < 0.3 else "🟡" if f1 < 0.7 else "🟢"
                    print(f"     {status} {entity_type:<15}: F1={f1:.4f}, N={support}")
        
        return seqeval_results
    
    def _compute_tag_level_metrics(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute tag-level metrics and per-entity tag analysis.
        
        🏷️ TAG-LEVEL ANALYSIS:
        ======================
        - Individual token classification accuracy
        - Per-entity type tag breakdown (B- vs I- performance)
        - Confusion matrix analysis
        - Training-relevant metrics
        """
        
        # Filter out ignored tokens (-100)
        mask = labels != -100
        valid_predictions = predictions[mask]
        valid_labels = labels[mask]
        
        # Overall tag accuracy
        tag_accuracy = np.mean(valid_predictions == valid_labels)
        
        # Per-entity tag analysis
        entity_tag_metrics = {}
        
        for entity_type in self.entity_types:
            b_tag_id = self.label2id.get(f'B-{entity_type}')
            i_tag_id = self.label2id.get(f'I-{entity_type}')
            
            if b_tag_id is not None and i_tag_id is not None:
                # B- tag metrics
                b_true_mask = valid_labels == b_tag_id
                b_pred_mask = valid_predictions == b_tag_id
                
                b_true_count = np.sum(b_true_mask)
                b_pred_count = np.sum(b_pred_mask)
                b_correct_count = np.sum(b_true_mask & b_pred_mask)
                
                # I- tag metrics  
                i_true_mask = valid_labels == i_tag_id
                i_pred_mask = valid_predictions == i_tag_id
                
                i_true_count = np.sum(i_true_mask)
                i_pred_count = np.sum(i_pred_mask)
                i_correct_count = np.sum(i_true_mask & i_pred_mask)
                
                # Store metrics
                entity_tag_metrics[entity_type] = {
                    'b_tag_metrics': {
                        'true': int(b_true_count),
                        'pred': int(b_pred_count), 
                        'correct': int(b_correct_count),
                        'accuracy': float(b_correct_count / max(b_true_count, 1))
                    },
                    'i_tag_metrics': {
                        'true': int(i_true_count),
                        'pred': int(i_pred_count),
                        'correct': int(i_correct_count), 
                        'accuracy': float(i_correct_count / max(i_true_count, 1))
                    }
                }
        
        # Compute micro and macro F1 at tag level
        all_entity_true = sum(
            metrics['b_tag_metrics']['true'] + metrics['i_tag_metrics']['true']
            for metrics in entity_tag_metrics.values()
        )
        all_entity_pred = sum(
            metrics['b_tag_metrics']['pred'] + metrics['i_tag_metrics']['pred']
            for metrics in entity_tag_metrics.values()
        )
        all_entity_correct = sum(
            metrics['b_tag_metrics']['correct'] + metrics['i_tag_metrics']['correct']
            for metrics in entity_tag_metrics.values()
        )
        
        # Micro F1 (frequency weighted)
        if all_entity_true > 0 and all_entity_pred > 0:
            micro_precision = all_entity_correct / all_entity_pred
            micro_recall = all_entity_correct / all_entity_true
            micro_f1 = (2 * micro_precision * micro_recall / 
                       (micro_precision + micro_recall) if 
                       (micro_precision + micro_recall) > 0 else 0)
        else:
            micro_f1 = 0.0
        
        # Macro F1 (unweighted average)
        entity_f1_scores = []
        for entity_type in self.entity_types:
            if entity_type in entity_tag_metrics:
                metrics = entity_tag_metrics[entity_type]
                
                total_true = (metrics['b_tag_metrics']['true'] + 
                             metrics['i_tag_metrics']['true'])
                total_pred = (metrics['b_tag_metrics']['pred'] + 
                             metrics['i_tag_metrics']['pred'])
                total_correct = (metrics['b_tag_metrics']['correct'] + 
                                metrics['i_tag_metrics']['correct'])
                
                if total_pred > 0 and total_true > 0:
                    precision = total_correct / total_pred
                    recall = total_correct / total_true
                    f1 = (2 * precision * recall / (precision + recall) 
                          if (precision + recall) > 0 else 0)
                    entity_f1_scores.append(f1)
        
        macro_f1 = np.mean(entity_f1_scores) if entity_f1_scores else 0.0
        
        if verbose:
            print(f"\n🏷️  TAG-LEVEL METRICS:")
            print("-" * 40)
            print(f"   Overall Tag Accuracy: {tag_accuracy:.4f}")
            print(f"   Micro F1 (tag-weighted): {micro_f1:.4f}")
            print(f"   Macro F1 (entity-avg): {macro_f1:.4f}")
        
        return {
            'tag_accuracy': tag_accuracy,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'entity_tag_metrics': entity_tag_metrics,
            'total_valid_tokens': len(valid_labels)
        }
    
    def _validate_bio_consistency(
        self, 
        predictions: List[List[str]],
        verbose: bool = True
    ) -> Dict[str, int]:
        """
        Validate BIO tagging consistency in predictions.
        
        🔍 BIO VALIDATION RULES:
        =======================
        - I-tag must be preceded by B-tag of same entity type
        - I-tag cannot appear without corresponding B-tag
        - Entity type must be consistent within entity span
        """
        
        total_sequences = len(predictions)
        invalid_sequences = 0
        orphaned_i_tags = 0
        total_violations = 0
        
        for seq_idx, sequence in enumerate(predictions):
            sequence_valid = True
            
            for i, tag in enumerate(sequence):
                if tag.startswith('I-'):
                    entity_type = tag[2:]
                    
                    # Look for valid B- tag before this I- tag
                    valid_start = False
                    
                    for j in range(i-1, -1, -1):
                        prev_tag = sequence[j]
                        
                        if prev_tag == 'O':
                            # Found O before I-, invalid
                            break
                        elif prev_tag.startswith('B-') and prev_tag[2:] == entity_type:
                            # Found matching B- tag
                            valid_start = True
                            break
                        elif prev_tag.startswith('I-') and prev_tag[2:] == entity_type:
                            # Found matching I- tag, continue searching
                            continue
                        else:
                            # Found different entity type, invalid
                            break
                    
                    if not valid_start:
                        orphaned_i_tags += 1
                        sequence_valid = False
                        total_violations += 1
            
            if not sequence_valid:
                invalid_sequences += 1
        
        if verbose:
            print(f"\n🔗 BIO CONSISTENCY VALIDATION:")
            print("-" * 40)
            print(f"   Total sequences: {total_sequences}")
            print(f"   Invalid sequences: {invalid_sequences} ({invalid_sequences/total_sequences*100:.1f}%)")
            print(f"   Orphaned I- tags: {orphaned_i_tags}")
        
        return {
            'total_sequences': total_sequences,
            'invalid_sequences': invalid_sequences,
            'orphaned_i_tags': orphaned_i_tags,
            'total_violations': total_violations
        }
    
    def _analyze_entity_boundaries(
        self, 
        predictions: List[List[str]], 
        references: List[List[str]],
        verbose: bool = True
    ) -> Dict[str, int]:
        """
        Analyze entity boundary detection accuracy.
        
        🎯 BOUNDARY ANALYSIS:
        ====================
        - Correct entity start detection (B- tag placement)
        - Correct entity end detection (transition to O or different entity)
        - Partial entity matches (correct start but wrong end)
        - Entity span accuracy
        """
        
        def extract_entity_spans(sequence):
            """Extract entity spans as (start, end, type) tuples."""
            entities = []
            current_entity = None
            
            for i, tag in enumerate(sequence):
                if tag.startswith('B-'):
                    # Start new entity
                    if current_entity is not None:
                        # End previous entity
                        entities.append((current_entity[0], i-1, current_entity[1]))
                    current_entity = (i, tag[2:])  # (start, type)
                elif tag.startswith('I-'):
                    # Continue entity
                    entity_type = tag[2:]
                    if current_entity is None or current_entity[1] != entity_type:
                        # Orphaned I- tag or type mismatch
                        if current_entity is not None:
                            entities.append((current_entity[0], i-1, current_entity[1]))
                        current_entity = (i, entity_type)  # Treat as new entity start
                elif tag == 'O':
                    # End current entity
                    if current_entity is not None:
                        entities.append((current_entity[0], i-1, current_entity[1]))
                        current_entity = None
            
            # Handle entity at end of sequence
            if current_entity is not None:
                entities.append((current_entity[0], len(sequence)-1, current_entity[1]))
            
            return entities
        
        total_entities_true = 0
        total_entities_pred = 0
        correct_boundaries = 0
        correct_starts = 0
        correct_ends = 0
        boundary_errors = 0
        
        for pred_seq, true_seq in zip(predictions, references):
            true_entities = extract_entity_spans(true_seq)
            pred_entities = extract_entity_spans(pred_seq)
            
            total_entities_true += len(true_entities)
            total_entities_pred += len(pred_entities)
            
            # Convert to sets for easier comparison
            true_set = set(true_entities)
            pred_set = set(pred_entities)
            
            # Exact boundary matches
            exact_matches = true_set & pred_set
            correct_boundaries += len(exact_matches)
            
            # Start position accuracy
            true_starts = {(start, entity_type) for start, _, entity_type in true_entities}
            pred_starts = {(start, entity_type) for start, _, entity_type in pred_entities}
            correct_starts += len(true_starts & pred_starts)
            
            # End position accuracy  
            true_ends = {(end, entity_type) for _, end, entity_type in true_entities}
            pred_ends = {(end, entity_type) for _, end, entity_type in pred_entities}
            correct_ends += len(true_ends & pred_ends)
        
        # Calculate boundary errors
        boundary_errors = total_entities_true - correct_boundaries
        
        if verbose:
            print(f"\n🎯 ENTITY BOUNDARY ANALYSIS:")
            print("-" * 40)
            print(f"   True entities: {total_entities_true}")
            print(f"   Predicted entities: {total_entities_pred}")
            print(f"   Exact boundary matches: {correct_boundaries}")
            
            if total_entities_true > 0:
                start_acc = correct_starts / total_entities_true
                end_acc = correct_ends / total_entities_true  
                boundary_acc = correct_boundaries / total_entities_true
                
                print(f"   Start accuracy: {start_acc:.4f}")
                print(f"   End accuracy: {end_acc:.4f}")
                print(f"   Complete boundary accuracy: {boundary_acc:.4f}")
        
        return {
            'total_entities_true': total_entities_true,
            'total_entities_pred': total_entities_pred,
            'correct_boundaries': correct_boundaries,
            'correct_starts': correct_starts,
            'correct_ends': correct_ends,
            'boundary_errors': boundary_errors
        }
    
    def _compute_crf_metrics(
        self, 
        model: Any, 
        predictions: np.ndarray, 
        labels: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute CRF-specific metrics and transition analysis.
        
        🔧 CRF ANALYSIS:
        ===============
        - Transition matrix analysis
        - Valid vs invalid transition rates
        - CRF constraint effectiveness
        - Viterbi decoding quality
        """
        
        crf_metrics = {}
        
        if hasattr(model, 'get_crf_transitions'):
            try:
                transitions = model.get_crf_transitions()
                
                # Analyze transition matrix
                if transitions is not None:
                    # Convert to numpy for analysis
                    if hasattr(transitions, 'detach'):
                        transitions_np = transitions.detach().cpu().numpy()
                    else:
                        transitions_np = np.array(transitions)
                    
                    # Compute transition statistics
                    transition_variance = np.var(transitions_np)
                    transition_range = np.max(transitions_np) - np.min(transitions_np)
                    
                    crf_metrics.update({
                        'transition_variance': float(transition_variance),
                        'transition_range': float(transition_range),
                        'crf_enabled': True
                    })
                    
                    if verbose:
                        print(f"\n🔧 CRF TRANSITION ANALYSIS:")
                        print("-" * 40)
                        print(f"   Transition matrix shape: {transitions_np.shape}")
                        print(f"   Transition variance: {transition_variance:.4f}")
                        print(f"   Transition range: {transition_range:.4f}")
                
            except Exception as e:
                if verbose:
                    print(f"\n⚠️  CRF analysis failed: {e}")
                crf_metrics['crf_analysis_error'] = str(e)
        
        # Sequence coherence analysis (applicable to CRF)
        coherence_score = self._compute_sequence_coherence(predictions, labels)
        crf_metrics['sequence_coherence'] = coherence_score
        
        return crf_metrics
    
    def _compute_sequence_coherence(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray
    ) -> float:
        """
        Compute sequence coherence score.
        
        Measures how well the predicted sequences follow BIO constraints
        compared to a naive tag-by-tag prediction.
        """
        
        # Convert to string format for analysis
        true_labels_str, pred_labels_str = self._convert_to_string_labels(
            predictions, labels
        )
        
        valid_transitions = 0
        total_transitions = 0
        
        for sequence in pred_labels_str:
            for i in range(len(sequence) - 1):
                current_tag = sequence[i]
                next_tag = sequence[i + 1]
                
                total_transitions += 1
                
                # Check if transition is valid according to BIO rules
                if self._is_valid_transition(current_tag, next_tag):
                    valid_transitions += 1
        
        coherence_score = valid_transitions / max(total_transitions, 1)
        return coherence_score
    
    def _is_valid_transition(self, from_tag: str, to_tag: str) -> bool:
        """
        Check if a transition between tags is valid according to BIO rules.
        """
        
        # O can transition to anything
        if from_tag == 'O':
            return True
        
        # B- can transition to I- of same type or O or any B-
        if from_tag.startswith('B-'):
            entity_type = from_tag[2:]
            if to_tag == 'O':
                return True
            elif to_tag.startswith('B-'):
                return True
            elif to_tag.startswith('I-') and to_tag[2:] == entity_type:
                return True
            else:
                return False
        
        # I- can transition to I- of same type or O or any B-
        if from_tag.startswith('I-'):
            entity_type = from_tag[2:]
            if to_tag == 'O':
                return True
            elif to_tag.startswith('B-'):
                return True
            elif to_tag.startswith('I-') and to_tag[2:] == entity_type:
                return True
            else:
                return False
        
        # Default: invalid
        return False
    
    def _aggregate_results(
        self, 
        seqeval_results: Dict[str, Any],
        tag_metrics: Dict[str, Any],
        bio_validation: Dict[str, int], 
        boundary_analysis: Dict[str, int],
        crf_metrics: Optional[Dict[str, float]]
    ) -> EvaluationResults:
        """
        Aggregate all computed metrics into a comprehensive results object.
        """
        
        results = EvaluationResults()
        
        # Overall metrics from SeqEval
        results.overall_f1 = seqeval_results['overall_f1']
        results.overall_precision = seqeval_results['overall_precision']
        results.overall_recall = seqeval_results['overall_recall']
        results.overall_accuracy = seqeval_results['overall_accuracy']
        
        # Aggregated metrics
        results.macro_f1 = tag_metrics['macro_f1']
        results.micro_f1 = tag_metrics['micro_f1']
        results.tag_level_accuracy = tag_metrics['tag_accuracy']
        
        # Sequence validation
        results.total_sequences = bio_validation['total_sequences']
        results.invalid_sequences = bio_validation['invalid_sequences']
        results.orphaned_i_tags = bio_validation['orphaned_i_tags']
        
        # Boundary analysis
        results.total_entities_true = boundary_analysis['total_entities_true']
        results.total_entities_pred = boundary_analysis['total_entities_pred']
        results.boundary_errors = boundary_analysis['boundary_errors']
        
        # Per-entity metrics
        entity_metrics = {}
        for entity_type in self.entity_types:
            if entity_type in seqeval_results:
                seqeval_entity = seqeval_results[entity_type]
                
                entity_metric = EntityMetrics(entity_type=entity_type)
                entity_metric.f1 = seqeval_entity.get('f1', 0.0)
                entity_metric.precision = seqeval_entity.get('precision', 0.0)
                entity_metric.recall = seqeval_entity.get('recall', 0.0)
                entity_metric.support = seqeval_entity.get('number', 0)
                
                # Add tag-level metrics if available
                if entity_type in tag_metrics['entity_tag_metrics']:
                    tag_data = tag_metrics['entity_tag_metrics'][entity_type]
                    entity_metric.b_tag_metrics = tag_data['b_tag_metrics']
                    entity_metric.i_tag_metrics = tag_data['i_tag_metrics']
                
                entity_metrics[entity_type] = entity_metric
        
        results.entity_metrics = entity_metrics
        
        # CRF metrics
        if crf_metrics:
            results.crf_metrics = crf_metrics
            results.sequence_coherence_score = crf_metrics.get('sequence_coherence', 0.0)
        
        return results
    
    def _print_results_summary(self, results: EvaluationResults):
        """Print a comprehensive summary of evaluation results."""
        
        print(f"\n📊 EVALUATION RESULTS SUMMARY:")
        print("="*60)
        
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Entity-level F1: {results.overall_f1:.4f}")
        print(f"   Precision: {results.overall_precision:.4f}")
        print(f"   Recall: {results.overall_recall:.4f}")
        
        print(f"\n📈 AGGREGATED METRICS:")
        print(f"   Macro F1 (entity-avg): {results.macro_f1:.4f}")
        print(f"   Micro F1 (tag-weighted): {results.micro_f1:.4f}")
        print(f"   Tag Accuracy: {results.tag_level_accuracy:.4f}")
        
        print(f"\n🔍 SEQUENCE ANALYSIS:")
        print(f"   Total sequences: {results.total_sequences:,}")
        print(f"   Invalid sequences: {results.invalid_sequences} ({results.invalid_sequences/results.total_sequences*100:.1f}%)")
        print(f"   Orphaned I- tags: {results.orphaned_i_tags}")
        
        print(f"\n🎯 ENTITY ANALYSIS:")
        print(f"   True entities: {results.total_entities_true:,}")
        print(f"   Predicted entities: {results.total_entities_pred:,}")
        print(f"   Boundary errors: {results.boundary_errors}")
        
        if results.crf_metrics:
            print(f"\n🔧 CRF ANALYSIS:")
            coherence = results.sequence_coherence_score
            print(f"   Sequence coherence: {coherence:.4f}")
        
        print("="*60)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_metrics_with_model_info(
    eval_pred, 
    label_list: List[str], 
    model=None, 
    use_crf: bool = False,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Convenience function for computing metrics in Trainer.
    
    This function is designed to be used as the compute_metrics function
    in HuggingFace Trainer.
    
    Args:
        eval_pred: Tuple of (predictions, labels) from Trainer
        label_list: List of all possible labels
        model: Model instance (for CRF analysis)
        use_crf: Whether CRF is being used
        verbose: Whether to print detailed analysis
    
    Returns:
        Dict: Metrics dictionary suitable for Trainer logging
    """
    
    predictions, labels = eval_pred
    
    if verbose:
        print(f"\n🧠 [CRF COMPATIBILITY] Computing metrics")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Use CRF: {use_crf}")
        print(f"   Model type: {type(model).__name__ if model else 'None'}")
    
    # Enhanced prediction format handling for CRF compatibility
    try:
        # CRF models can return different prediction formats
        if len(predictions.shape) == 3:
            if verbose:
                print(f"   🔄 Converting 3D predictions (logits) to 2D using argmax")
                print(f"   Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            
            # Validate that we have valid logits before argmax
            if predictions.size == 0:
                raise ValueError("Empty predictions array - cannot compute argmax")
            
            # Check for any empty sequences that would cause argmax to fail
            for i, seq in enumerate(predictions):
                if seq.size == 0:
                    raise ValueError(f"Empty sequence at index {i} - cannot compute argmax")
            
            predictions = np.argmax(predictions, axis=2)
            
        elif len(predictions.shape) == 2:
            if verbose:
                print(f"   ✅ Using 2D predictions directly (already decoded)")
                print(f"   Prediction range: [{predictions.min()}, {predictions.max()}]")
        else:
            raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
        
        # Validate final predictions
        if predictions.size == 0:
            raise ValueError("Final predictions array is empty")
        
        # Validate that predictions are within valid label range
        max_label_id = len(label_list) - 1
        if predictions.max() > max_label_id:
            print(f"   ⚠️  Warning: Max prediction ID ({predictions.max()}) exceeds max label ID ({max_label_id})")
            # Clip invalid predictions to valid range
            predictions = np.clip(predictions, 0, max_label_id)
        
        if verbose:
            print(f"   ✅ Final predictions shape: {predictions.shape}")
            print(f"   ✅ Prediction range: [{predictions.min()}, {predictions.max()}]")
            print(f"   ✅ Valid label range: [0, {max_label_id}]")
        
    except Exception as e:
        print(f"   ❌ Error in prediction processing: {e}")
        print(f"   ❌ Predictions info: shape={predictions.shape}, dtype={predictions.dtype}")
        if predictions.size > 0:
            print(f"   ❌ Sample predictions: {predictions.flat[:10]}")
        raise
    
    # Initialize calculator
    calculator = NERMetricsCalculator(label_list, use_crf)
    
    # Compute comprehensive metrics
    try:
        results = calculator.compute_comprehensive_metrics(
            predictions, labels, model, verbose
        )
        
        if verbose:
            print(f"   ✅ Metrics computation successful")
        
        # Return summary suitable for Trainer
        return results.get_summary_dict()
        
    except Exception as e:
        print(f"   ❌ Error in metrics calculation: {e}")
        raise


def export_detailed_results(
    results: EvaluationResults, 
    output_file: str,
    format: str = "json"
):
    """
    Export detailed evaluation results to file.
    
    Args:
        results: EvaluationResults object
        output_file: Output file path
        format: Export format ("json", "csv", "txt")
    """
    
    if format == "json":
        summary = results.get_summary_dict()
        
        # Add detailed entity metrics
        detailed_export = {
            'summary': summary,
            'entity_details': {}
        }
        
        for entity_type, metrics in results.entity_metrics.items():
            detailed_export['entity_details'][entity_type] = {
                'f1': metrics.f1,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'support': metrics.support,
                'b_tag_metrics': metrics.b_tag_metrics,
                'i_tag_metrics': metrics.i_tag_metrics,
                'combined_f1': metrics.get_combined_f1()
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            clean_detailed_export = clean_for_json_serialization(detailed_export)
            json.dump(clean_detailed_export, f, indent=2, ensure_ascii=False)
    
    elif format == "csv":
        import pandas as pd
        
        # Create CSV with per-entity metrics
        entity_data = []
        for entity_type, metrics in results.entity_metrics.items():
            entity_data.append({
                'entity_type': entity_type,
                'f1': metrics.f1,
                'precision': metrics.precision, 
                'recall': metrics.recall,
                'support': metrics.support,
                'b_accuracy': metrics.b_tag_metrics['accuracy'],
                'i_accuracy': metrics.i_tag_metrics['accuracy'],
                'combined_f1': metrics.get_combined_f1()
            })
        
        df = pd.DataFrame(entity_data)
        df.to_csv(output_file, index=False)
    
    elif format == "txt":
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("GREEK LEGAL NER - DETAILED EVALUATION RESULTS\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Overall F1: {results.overall_f1:.4f}\n")
            f.write(f"Overall Precision: {results.overall_precision:.4f}\n")
            f.write(f"Overall Recall: {results.overall_recall:.4f}\n\n")
            
            f.write("Per-Entity Results:\n")
            f.write("-"*30 + "\n")
            
            for entity_type, metrics in results.entity_metrics.items():
                f.write(f"{entity_type}:\n")
                f.write(f"  F1: {metrics.f1:.4f}\n")
                f.write(f"  Precision: {metrics.precision:.4f}\n") 
                f.write(f"  Recall: {metrics.recall:.4f}\n")
                f.write(f"  Support: {metrics.support}\n\n")


# Export main classes and functions
__all__ = [
    'EntityMetrics',
    'EvaluationResults', 
    'NERMetricsCalculator',
    'compute_metrics_with_model_info',
    'export_detailed_results'
]
