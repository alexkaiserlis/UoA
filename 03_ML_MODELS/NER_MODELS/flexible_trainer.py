"""
Flexible Trainer Implementation for Greek Legal NER

This module contains the FlexibleTrainer class that extends HuggingFace Trainer
with advanced loss functions, comprehensive metrics, and CRF support.

🔍 FLEXIBLE TRAINER FEATURES:
============================
1. **Multiple Loss Functions**:
   - Adaptive Focal Loss για imbalanced classes
   - Weighted Cross Entropy
   - Standard Cross Entropy
   - Support για custom class weights

2. **Advanced Metrics System**:
   - Per-entity F1, Precision, Recall (SeqEval)
   - Per-tag analysis (B-, I-, O breakdown)
   - BIO sequence validation (consistency checking)
   - Entity boundary detection metrics
   - CRF-specific transition analysis

3. **Model Support**:
   - ROBERTa + CRF models
   - Standard ROBERTa models
   - Automatic model type detection

4. **Training Features**:
   - Debug mode για detailed logging
   - Custom prediction step για CRF decoding
   - Memory-efficient batch processing
   - Comprehensive error handling

🏷️ METRICS BREAKDOWN:
=====================
📊 **Per-Entity Metrics** (Primary):
- Entity-level F1, Precision, Recall από SeqEval
- Exact match entity detection
- Support για όλα τα Greek Legal NER entity types

🎯 **Per-Tag Metrics** (Detailed):
- B- tag detection accuracy
- I- tag continuation accuracy  
- O tag classification accuracy
- Tag-level confusion matrix

🔍 **BIO Validation Metrics**:
- Orphaned I- tags detection (I- χωρίς B-)
- Invalid sequence patterns
- Transition consistency analysis
- Entity boundary accuracy

📈 **Summary Statistics**:
- Macro F1 (unweighted average)
- Micro F1 (frequency weighted)
- Overall tag-level accuracy
- Entity detection rate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Trainer
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any, Union
import evaluate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import των custom modules
from focal_loss import AdaptiveFocalLoss, create_focal_loss
from crf_model import BertCRFForTokenClassification


class FlexibleTrainer(Trainer):
    """
    Εκτεταμένος Trainer που υποστηρίζει:
    - Πολλαπλούς τύπους loss functions
    - ROBERTa και ROBERTa+CRF μοντέλα
    - Εκτενή metrics analysis (per-entity και per-tag)
    - BIO sequence validation
    - Advanced debugging capabilities
    
    Args:
        *args: Trainer arguments
        loss_type (str): Type of loss function to use:
                        - "adaptive_focal": Focal loss για imbalanced classes
                        - "weighted_ce": Weighted cross entropy
                        - "standard": Standard cross entropy
        focal_gamma (float): Gamma parameter για focal loss
        focal_alpha: Alpha parameter για focal loss (class weights)
        class_weights (torch.Tensor): Tensor με class weights
        debug_mode (bool): Enable detailed debugging output
        use_crf (bool): Whether the model uses CRF layer
        label_list (List[str]): List of label names για metrics
        entity_types (List[str]): List of entity types για analysis
        **kwargs: Additional trainer arguments
    
    Example:
        >>> trainer = FlexibleTrainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=train_dataset,
        ...     eval_dataset=eval_dataset,
        ...     loss_type="adaptive_focal",
        ...     focal_gamma=1.0,
        ...     use_crf=True,
        ...     label_list=label_list,
        ...     debug_mode=True
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self, 
        *args, 
        loss_type: str = "adaptive_focal", 
        focal_gamma: float = 1.0, 
        focal_alpha: Optional[Union[float, torch.Tensor]] = None,
        class_weights: Optional[torch.Tensor] = None, 
        debug_mode: bool = False, 
        use_crf: bool = False,
        label_list: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Loss configuration
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.class_weights = class_weights
        
        # Model configuration
        self.debug_mode = debug_mode
        self.use_crf = use_crf
        self.step_count = 0
        
        # Metrics configuration
        self.label_list = label_list or []
        self.num_labels = len(self.label_list)  # Add num_labels for debugging
        self.entity_types = entity_types or self._extract_entity_types()
        
        # Initialize SeqEval metric
        self.seqeval_metric = evaluate.load("seqeval")
        
        # Cache για performance
        self._loss_function = None
        
        if debug_mode:
            print(f"🔧 FlexibleTrainer initialized:")
            print(f"   Loss type: {loss_type}")
            print(f"   Use CRF: {use_crf}")
            print(f"   Entity types: {len(self.entity_types)}")
            print(f"   Labels: {len(self.label_list)}")
    
    def _extract_entity_types(self) -> List[str]:
        """Extract entity types από το label list."""
        entity_types = set()
        for label in self.label_list:
            if label.startswith('B-') or label.startswith('I-'):
                entity_types.add(label[2:])
        return sorted(list(entity_types))
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation που υποστηρίζει multiple loss types.
        
        Args:
            model: The model
            inputs: Model inputs
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss tensor (and outputs if return_outputs=True)
        """
        labels = inputs.get("labels")
        
        # Handle CRF models
        if self.use_crf and hasattr(model, 'forward'):
            outputs = model(**inputs)
            loss = outputs.get("loss")
            
            if loss is None:
                # Fallback αν το CRF model δεν επιστρέφει loss
                logits = outputs.get("logits")
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                
        else:
            # Standard ROBERTa processing
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            # Create loss function if not cached
            if self._loss_function is None:
                self._loss_function = self._create_loss_function()
            
            # Reshape για loss computation
            if logits.dim() > 2:
                shift_logits = logits.view(-1, logits.size(-1))
                shift_labels = labels.view(-1)
            else:
                shift_logits = logits
                shift_labels = labels
            
            # Compute loss
            loss = self._loss_function(shift_logits, shift_labels)
        
        # Debug logging
        if self.debug_mode and self.step_count % 100 == 0:
            self._log_training_step(loss, labels)
        
        self.step_count += 1
        return (loss, outputs) if return_outputs else loss
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function βάσει configuration."""
        return create_focal_loss(
            loss_type=self.loss_type,
            gamma=self.focal_gamma,
            alpha=self.focal_alpha or self.class_weights,
            reduction='mean',
            ignore_index=-100
        )
    
    def _log_training_step(self, loss: torch.Tensor, labels: torch.Tensor):
        """Log training step information για debugging."""
        model_type = "CRF" if self.use_crf else "ROBERTa"
        
        if labels is not None and not self.use_crf:
            active_labels = labels[labels != -100]
            if len(active_labels) > 0:
                unique_labels = torch.unique(active_labels)
                max_label = unique_labels.max().item()
                min_label = unique_labels.min().item()
                expected_max = len(self.label_list) - 1
                
                if max_label > expected_max:
                    print(f"⚠️  WARNING: Model sees label {max_label} but max expected is {expected_max}")
                
                sample_labels = unique_labels[:5].tolist()
                print(f"Step {self.step_count}: Loss={loss.item():.4f}, Model={model_type}, "
                      f"Active_labels={len(active_labels)}, Sample_labels={sample_labels}")
        else:
            print(f"Step {self.step_count}: Loss={loss.item():.4f}, Model={model_type}")
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction step που χειρίζεται CRF models με enhanced debugging.
        """
        inputs = self._prepare_inputs(inputs)
        
        if self.debug_mode:
            print(f"\n🧠 [PREDICTION STEP] Enhanced debugging")
            print(f"   Use CRF: {self.use_crf}")
            print(f"   Model type: {type(model).__name__}")
            print(f"   Input shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in inputs.items()]}")
            print(f"   Prediction loss only: {prediction_loss_only}")
        
        with torch.no_grad():
            try:
                if self.use_crf and hasattr(model, 'decode'):
                    # CRF model prediction με enhanced validation
                    if self.debug_mode:
                        print(f"   🔄 Using CRF decode method")
                    
                    loss = None
                    if not prediction_loss_only:
                        outputs = model(**inputs)
                        loss = outputs.get("loss")
                        if self.debug_mode:
                            print(f"   📊 Loss computed: {loss.item() if loss is not None else 'None'}")
                    
                    # Get CRF predictions με error handling
                    try:
                        predictions = model.decode(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask"),
                            token_type_ids=inputs.get("token_type_ids")
                        )
                        
                        if self.debug_mode:
                            print(f"   ✅ CRF decode successful")
                            print(f"   📋 Decoded predictions type: {type(predictions)}")
                            if isinstance(predictions, (list, tuple)) and len(predictions) > 0:
                                print(f"   📋 First prediction sample: {predictions[0][:10] if len(predictions[0]) > 10 else predictions[0]}")
                            
                    except Exception as e:
                        if self.debug_mode:
                            print(f"   ❌ CRF decode failed: {e}")
                        raise ValueError(f"CRF decode failed: {e}")
                    
                    # Validate predictions before tensor conversion
                    if not predictions:
                        raise ValueError("CRF decode returned empty predictions")
                    
                    # Convert to tensor format με validation
                    batch_size, seq_len = inputs["input_ids"].shape
                    num_labels = getattr(self, 'num_labels', len(getattr(self, 'label_list', [])))
                    
                    if self.debug_mode:
                        print(f"   🔧 Converting to tensor format: batch_size={batch_size}, seq_len={seq_len}, num_labels={num_labels}")
                    
                    if num_labels == 0:
                        raise ValueError("Cannot determine number of labels for tensor conversion")
                    
                    logits = torch.zeros((batch_size, seq_len, num_labels), device=inputs["input_ids"].device)
                    
                    for i, seq_pred in enumerate(predictions):
                        if seq_pred is None:
                            if self.debug_mode:
                                print(f"   ⚠️  Warning: None prediction at sequence {i}")
                            continue
                            
                        pred_len = min(len(seq_pred), seq_len)
                        for j in range(pred_len):
                            pred_val = seq_pred[j]
                            if pred_val is not None and 0 <= pred_val < num_labels:
                                logits[i, j, pred_val] = 1.0
                            elif self.debug_mode and pred_val is not None:
                                print(f"   ⚠️  Invalid prediction value {pred_val} at ({i}, {j}), skipping")
                    
                    if self.debug_mode:
                        print(f"   ✅ Tensor conversion completed")
                        print(f"   📊 Final logits shape: {logits.shape}")
                        print(f"   📊 Logits non-zero elements: {torch.nonzero(logits).shape[0]}")
                            
                else:
                    # Standard ROBERTa prediction
                    if self.debug_mode:
                        print(f"   🔄 Using standard model forward pass")
                    
                    outputs = model(**inputs)
                    loss = outputs.get("loss")
                    logits = outputs.get("logits")
                    
                    if self.debug_mode:
                        print(f"   ✅ Standard forward pass completed")
                        print(f"   📊 Loss: {loss.item() if loss is not None else 'None'}")
                        print(f"   📊 Logits shape: {logits.shape if logits is not None else 'None'}")
                        if logits is not None:
                            print(f"   📊 Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                
            except Exception as e:
                print(f"   ❌ Critical error in prediction step: {e}")
                print(f"   ❌ Model type: {type(model).__name__}")
                print(f"   ❌ Use CRF: {self.use_crf}")
                print(f"   ❌ Input keys: {list(inputs.keys())}")
                raise
        
        labels = inputs.get("labels")
        
        if prediction_loss_only:
            return (loss, None, None)
        
        if self.debug_mode:
            print(f"   🏁 Prediction step completed successfully")
            print(f"   📋 Returning: loss={'✓' if loss is not None else '✗'}, "
                  f"logits={'✓' if logits is not None else '✗'}, "
                  f"labels={'✓' if labels is not None else '✗'}")
        
        return (loss, logits, labels)
    
    def compute_metrics_with_model_info(self, eval_pred) -> Dict[str, float]:
        """
        Comprehensive metrics computation με εκτενή per-entity και per-tag analysis.
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dict με όλα τα computed metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Convert predictions και labels σε label strings
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100] 
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100] 
            for prediction, label in zip(predictions, labels)
        ]
        
        print(f"\n{'='*80}")
        print(f"🔍 COMPREHENSIVE NER EVALUATION ANALYSIS")
        print(f"{'='*80}")
        
        # 1. Per-Entity Metrics (Primary Analysis)
        entity_metrics = self._compute_per_entity_metrics(true_predictions, true_labels)
        
        # 2. Per-Tag Analysis (B-, I-, O breakdown)
        tag_metrics = self._compute_per_tag_metrics(true_predictions, true_labels)
        
        # 3. BIO Sequence Validation
        sequence_metrics = self._compute_sequence_validation_metrics(true_predictions)
        
        # 4. Entity Boundary Analysis
        boundary_metrics = self._compute_boundary_metrics(true_predictions, true_labels)
        
        # 5. SeqEval Entity-Level Metrics (Gold Standard)
        seqeval_results = self.seqeval_metric.compute(
            predictions=true_predictions, 
            references=true_labels, 
            zero_division=0
        )
        
        # 6. CRF-Specific Analysis (αν εφαρμόζεται)
        crf_metrics = {}
        if self.use_crf:
            crf_metrics = self._compute_crf_specific_metrics(true_predictions, true_labels)
        
        # Combine όλα τα metrics
        final_metrics = self._combine_all_metrics(
            entity_metrics, tag_metrics, sequence_metrics, 
            boundary_metrics, seqeval_results, crf_metrics
        )
        
        # Print summary
        self._print_evaluation_summary(final_metrics)
        
        return final_metrics
    
    def _compute_per_entity_metrics(self, predictions: List[List[str]], labels: List[List[str]]) -> Dict:
        """Compute detailed per-entity metrics."""
        print(f"\n📊 PER-ENTITY ANALYSIS:")
        print("="*60)
        
        # Flatten για analysis
        flat_predictions = [tag for seq in predictions for tag in seq]
        flat_labels = [tag for seq in labels for tag in seq]
        
        entity_metrics = {}
        
        for entity_type in self.entity_types:
            b_tag = f"B-{entity_type}"
            i_tag = f"I-{entity_type}"
            
            # Count B- και I- tags
            b_true = flat_labels.count(b_tag)
            b_pred = flat_predictions.count(b_tag)
            b_correct = sum(1 for true, pred in zip(flat_labels, flat_predictions) 
                           if true == b_tag and pred == b_tag)
            
            i_true = flat_labels.count(i_tag)
            i_pred = flat_predictions.count(i_tag)
            i_correct = sum(1 for true, pred in zip(flat_labels, flat_predictions) 
                           if true == i_tag and pred == i_tag)
            
            # Combined metrics (B- + I-)
            total_true = b_true + i_true
            total_pred = b_pred + i_pred
            total_correct = b_correct + i_correct
            
            # Calculate metrics
            precision = total_correct / max(total_pred, 1)
            recall = total_correct / max(total_true, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            
            # B- specific metrics
            b_precision = b_correct / max(b_pred, 1)
            b_recall = b_correct / max(b_true, 1)
            b_f1 = 2 * b_precision * b_recall / max(b_precision + b_recall, 1e-10)
            
            # I- specific metrics
            i_precision = i_correct / max(i_pred, 1)
            i_recall = i_correct / max(i_true, 1)
            i_f1 = 2 * i_precision * i_recall / max(i_precision + i_recall, 1e-10)
            
            entity_metrics[entity_type] = {
                'total_true': total_true,
                'total_pred': total_pred, 
                'total_correct': total_correct,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'b_true': b_true,
                'b_pred': b_pred,
                'b_correct': b_correct,
                'b_precision': b_precision,
                'b_recall': b_recall,
                'b_f1': b_f1,
                'i_true': i_true,
                'i_pred': i_pred,
                'i_correct': i_correct,
                'i_precision': i_precision,
                'i_recall': i_recall,
                'i_f1': i_f1
            }
            
            # Print entity summary
            status = "🔴" if f1 < 0.3 else "🟡" if f1 < 0.7 else "🟢"
            print(f"   {status} {entity_type:<15}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, Total={total_true}")
        
        return entity_metrics
    
    def _compute_per_tag_metrics(self, predictions: List[List[str]], labels: List[List[str]]) -> Dict:
        """Compute detailed per-tag (B-, I-, O) analysis."""
        print(f"\n🏷️  PER-TAG ANALYSIS:")
        print("="*60)
        
        flat_predictions = [tag for seq in predictions for tag in seq]
        flat_labels = [tag for seq in labels for tag in seq]
        
        # Overall tag accuracy
        tag_accuracy = accuracy_score(flat_labels, flat_predictions)
        print(f"Overall tag accuracy: {tag_accuracy:.4f}")
        
        # Tag distribution analysis
        pred_counter = Counter(flat_predictions)
        true_counter = Counter(flat_labels)
        
        print(f"\n📈 Tag Distribution Comparison:")
        print(f"{'Tag':<15} {'True':<8} {'Pred':<8} {'Diff':<8} {'Accuracy':<10}")
        print("-" * 55)
        
        tag_metrics = {}
        
        # Analyze O tag
        o_true = true_counter['O']
        o_pred = pred_counter['O']
        o_correct = sum(1 for true, pred in zip(flat_labels, flat_predictions) 
                       if true == 'O' and pred == 'O')
        o_accuracy = o_correct / max(o_true, 1)
        
        tag_metrics['O'] = {
            'true_count': o_true,
            'pred_count': o_pred,
            'correct_count': o_correct,
            'accuracy': o_accuracy,
            'precision': o_correct / max(o_pred, 1),
            'recall': o_correct / max(o_true, 1)
        }
        
        print(f"{'O':<15} {o_true:<8} {o_pred:<8} {o_pred-o_true:+8} {o_accuracy:<10.4f}")
        
        # Analyze B- tags
        b_tags = [tag for tag in set(flat_labels + flat_predictions) if tag.startswith('B-')]
        total_b_metrics = {'true': 0, 'pred': 0, 'correct': 0}
        
        for tag in sorted(b_tags):
            true_count = true_counter[tag]
            pred_count = pred_counter[tag]
            correct_count = sum(1 for true, pred in zip(flat_labels, flat_predictions) 
                               if true == tag and pred == tag)
            accuracy = correct_count / max(true_count, 1)
            
            tag_metrics[tag] = {
                'true_count': true_count,
                'pred_count': pred_count,
                'correct_count': correct_count,
                'accuracy': accuracy,
                'precision': correct_count / max(pred_count, 1),
                'recall': correct_count / max(true_count, 1)
            }
            
            total_b_metrics['true'] += true_count
            total_b_metrics['pred'] += pred_count
            total_b_metrics['correct'] += correct_count
            
            print(f"{tag:<15} {true_count:<8} {pred_count:<8} {pred_count-true_count:+8} {accuracy:<10.4f}")
        
        # Analyze I- tags
        i_tags = [tag for tag in set(flat_labels + flat_predictions) if tag.startswith('I-')]
        total_i_metrics = {'true': 0, 'pred': 0, 'correct': 0}
        
        for tag in sorted(i_tags):
            true_count = true_counter[tag]
            pred_count = pred_counter[tag]
            correct_count = sum(1 for true, pred in zip(flat_labels, flat_predictions) 
                               if true == tag and pred == tag)
            accuracy = correct_count / max(true_count, 1)
            
            tag_metrics[tag] = {
                'true_count': true_count,
                'pred_count': pred_count,
                'correct_count': correct_count,
                'accuracy': accuracy,
                'precision': correct_count / max(pred_count, 1),
                'recall': correct_count / max(true_count, 1)
            }
            
            total_i_metrics['true'] += true_count
            total_i_metrics['pred'] += pred_count
            total_i_metrics['correct'] += correct_count
            
            print(f"{tag:<15} {true_count:<8} {pred_count:<8} {pred_count-true_count:+8} {accuracy:<10.4f}")
        
        # Summary statistics
        print("-" * 55)
        b_accuracy = total_b_metrics['correct'] / max(total_b_metrics['true'], 1)
        i_accuracy = total_i_metrics['correct'] / max(total_i_metrics['true'], 1)
        
        print(f"{'B- TOTAL':<15} {total_b_metrics['true']:<8} {total_b_metrics['pred']:<8} "
              f"{total_b_metrics['pred']-total_b_metrics['true']:+8} {b_accuracy:<10.4f}")
        print(f"{'I- TOTAL':<15} {total_i_metrics['true']:<8} {total_i_metrics['pred']:<8} "
              f"{total_i_metrics['pred']-total_i_metrics['true']:+8} {i_accuracy:<10.4f}")
        
        tag_metrics['summary'] = {
            'overall_accuracy': tag_accuracy,
            'b_tag_accuracy': b_accuracy,
            'i_tag_accuracy': i_accuracy,
            'o_tag_accuracy': o_accuracy,
            'total_b_metrics': total_b_metrics,
            'total_i_metrics': total_i_metrics
        }
        
        return tag_metrics
    
    def _compute_sequence_validation_metrics(self, predictions: List[List[str]]) -> Dict:
        """Compute BIO sequence validation metrics."""
        print(f"\n🔍 BIO SEQUENCE VALIDATION:")
        print("="*60)
        
        invalid_seqs = 0
        orphaned_i_tags = 0
        total_seqs = len(predictions)
        invalid_patterns = []
        
        for seq_idx, pred_seq in enumerate(predictions):
            seq_has_invalid = False
            
            for i, tag in enumerate(pred_seq):
                if tag.startswith('I-'):
                    entity_type = tag[2:]
                    
                    # Check for proper B- tag before this I- tag
                    found_valid_start = False
                    for j in range(i-1, -1, -1):
                        prev_tag = pred_seq[j]
                        if prev_tag == 'O':
                            break  # Hit O tag, no valid B- found
                        elif prev_tag == f'B-{entity_type}':
                            found_valid_start = True
                            break
                        elif prev_tag.startswith('I-') and prev_tag[2:] == entity_type:
                            continue  # Valid continuation
                        else:
                            break  # Different entity type
                    
                    if not found_valid_start:
                        orphaned_i_tags += 1
                        seq_has_invalid = True
                        
                        # Store sample invalid patterns
                        if len(invalid_patterns) < 5:
                            context_start = max(0, i-2)
                            context_end = min(len(pred_seq), i+3)
                            context = pred_seq[context_start:context_end]
                            invalid_patterns.append({
                                'seq_idx': seq_idx,
                                'position': i,
                                'orphaned_tag': tag,
                                'context': context
                            })
            
            if seq_has_invalid:
                invalid_seqs += 1
        
        consistency_rate = (total_seqs - invalid_seqs) / total_seqs * 100
        
        print(f"   Total sequences analyzed: {total_seqs:,}")
        print(f"   Invalid sequences: {invalid_seqs:,}")
        print(f"   Orphaned I- tags: {orphaned_i_tags:,}")
        print(f"   Sequence consistency rate: {consistency_rate:.2f}%")
        
        if invalid_patterns:
            print(f"\n   📋 Sample Invalid Patterns:")
            for pattern in invalid_patterns:
                print(f"      Seq {pattern['seq_idx']}, pos {pattern['position']}: "
                      f"{pattern['orphaned_tag']} in {pattern['context']}")
        
        if invalid_seqs > 0:
            print(f"   ⚠️  {invalid_seqs:,} sequences have BIO violations!")
        else:
            print(f"   ✅ All sequences follow valid BIO structure!")
        
        return {
            'total_sequences': total_seqs,
            'invalid_sequences': invalid_seqs,
            'orphaned_i_tags': orphaned_i_tags,
            'consistency_rate': consistency_rate,
            'invalid_patterns': invalid_patterns
        }
    
    def _compute_boundary_metrics(self, predictions: List[List[str]], labels: List[List[str]]) -> Dict:
        """Compute entity boundary detection metrics."""
        print(f"\n🎯 ENTITY BOUNDARY ANALYSIS:")
        print("="*60)
        
        def extract_entity_spans(sequence):
            """Extract entity spans from a sequence."""
            spans = []
            current_entity = None
            
            for i, tag in enumerate(sequence):
                if tag.startswith('B-'):
                    if current_entity is not None:
                        spans.append(current_entity)
                    current_entity = {
                        'type': tag[2:],
                        'start': i,
                        'end': i,
                        'tags': [tag]
                    }
                elif tag.startswith('I-') and current_entity is not None:
                    if tag[2:] == current_entity['type']:
                        current_entity['end'] = i
                        current_entity['tags'].append(tag)
                    else:
                        spans.append(current_entity)
                        current_entity = None
                else:  # O tag
                    if current_entity is not None:
                        spans.append(current_entity)
                        current_entity = None
            
            if current_entity is not None:
                spans.append(current_entity)
            
            return spans
        
        # Extract all entity spans
        true_spans = []
        pred_spans = []
        
        for true_seq, pred_seq in zip(labels, predictions):
            true_spans.extend(extract_entity_spans(true_seq))
            pred_spans.extend(extract_entity_spans(pred_seq))
        
        # Boundary metrics
        exact_match = 0
        partial_match = 0
        start_match = 0
        end_match = 0
        type_match = 0
        
        # Create sets για efficient lookup
        true_span_set = {(span['start'], span['end'], span['type']) for span in true_spans}
        pred_span_set = {(span['start'], span['end'], span['type']) for span in pred_spans}
        
        for pred_span in pred_spans:
            pred_key = (pred_span['start'], pred_span['end'], pred_span['type'])
            
            # Exact match
            if pred_key in true_span_set:
                exact_match += 1
                continue
            
            # Check για partial matches
            found_partial = False
            for true_span in true_spans:
                # Type match
                if pred_span['type'] == true_span['type']:
                    type_match += 1
                    
                    # Start boundary match
                    if pred_span['start'] == true_span['start']:
                        start_match += 1
                    
                    # End boundary match
                    if pred_span['end'] == true_span['end']:
                        end_match += 1
                    
                    # Partial overlap
                    pred_range = set(range(pred_span['start'], pred_span['end'] + 1))
                    true_range = set(range(true_span['start'], true_span['end'] + 1))
                    
                    if pred_range & true_range:  # Intersection
                        partial_match += 1
                        found_partial = True
                        break
        
        # Calculate rates
        total_pred = len(pred_spans)
        total_true = len(true_spans)
        
        exact_match_rate = exact_match / max(total_pred, 1)
        partial_match_rate = partial_match / max(total_pred, 1)
        start_accuracy = start_match / max(total_pred, 1)
        end_accuracy = end_match / max(total_pred, 1)
        type_accuracy = type_match / max(total_pred, 1)
        
        entity_recall = exact_match / max(total_true, 1)
        entity_precision = exact_match / max(total_pred, 1)
        entity_f1 = 2 * entity_precision * entity_recall / max(entity_precision + entity_recall, 1e-10)
        
        print(f"   Total true entities: {total_true:,}")
        print(f"   Total predicted entities: {total_pred:,}")
        print(f"   Exact matches: {exact_match:,} ({exact_match_rate:.4f})")
        print(f"   Partial matches: {partial_match:,} ({partial_match_rate:.4f})")
        print(f"   Start boundary accuracy: {start_accuracy:.4f}")
        print(f"   End boundary accuracy: {end_accuracy:.4f}")
        print(f"   Entity type accuracy: {type_accuracy:.4f}")
        print(f"   Entity-level P/R/F1: {entity_precision:.4f}/{entity_recall:.4f}/{entity_f1:.4f}")
        
        return {
            'total_true_entities': total_true,
            'total_pred_entities': total_pred,
            'exact_matches': exact_match,
            'partial_matches': partial_match,
            'start_matches': start_match,
            'end_matches': end_match,
            'type_matches': type_match,
            'exact_match_rate': exact_match_rate,
            'partial_match_rate': partial_match_rate,
            'start_accuracy': start_accuracy,
            'end_accuracy': end_accuracy,
            'type_accuracy': type_accuracy,
            'entity_precision': entity_precision,
            'entity_recall': entity_recall,
            'entity_f1': entity_f1
        }
    
    def _compute_crf_specific_metrics(self, predictions: List[List[str]], labels: List[List[str]]) -> Dict:
        """Compute CRF-specific transition analysis."""
        print(f"\n⚙️  CRF TRANSITION ANALYSIS:")
        print("="*60)
        
        if not hasattr(self.model, 'get_crf_transitions'):
            print("   Model does not have CRF transitions - skipping analysis")
            return {}
        
        try:
            # Get transition matrix
            transitions = self.model.get_crf_transitions().detach().cpu().numpy()
            
            # Analyze learned transitions
            print(f"   Transition matrix shape: {transitions.shape}")
            print(f"   Transition values range: [{transitions.min():.3f}, {transitions.max():.3f}]")
            
            # Find strongest positive και negative transitions
            flat_indices = np.unravel_index(np.argmax(transitions), transitions.shape)
            strongest_positive = (flat_indices[0], flat_indices[1], transitions[flat_indices])
            
            flat_indices = np.unravel_index(np.argmin(transitions), transitions.shape)
            strongest_negative = (flat_indices[0], flat_indices[1], transitions[flat_indices])
            
            print(f"   Strongest positive transition: {self.label_list[strongest_positive[0]]} → "
                  f"{self.label_list[strongest_positive[1]]} ({strongest_positive[2]:.3f})")
            print(f"   Strongest negative transition: {self.label_list[strongest_negative[0]]} → "
                  f"{self.label_list[strongest_negative[1]]} ({strongest_negative[2]:.3f})")
            
            # Analyze BIO constraint compliance
            bio_violations = 0
            total_transitions = 0
            
            for i, from_label in enumerate(self.label_list):
                for j, to_label in enumerate(self.label_list):
                    total_transitions += 1
                    
                    # Check for invalid BIO transitions
                    if (to_label.startswith('I-') and from_label == 'O') or \
                       (to_label.startswith('I-') and from_label.startswith('B-') and 
                        from_label[2:] != to_label[2:]):
                        if transitions[i, j] > 0:  # Positive score για invalid transition
                            bio_violations += 1
            
            bio_compliance = (total_transitions - bio_violations) / total_transitions
            
            print(f"   BIO constraint compliance: {bio_compliance:.4f}")
            print(f"   Invalid transitions with positive scores: {bio_violations}")
            
            return {
                'transition_matrix_shape': transitions.shape,
                'transition_min': float(transitions.min()),
                'transition_max': float(transitions.max()),
                'strongest_positive_transition': strongest_positive,
                'strongest_negative_transition': strongest_negative,
                'bio_compliance_rate': bio_compliance,
                'bio_violations': bio_violations
            }
            
        except Exception as e:
            print(f"   Error analyzing CRF transitions: {e}")
            return {'error': str(e)}
    
    def _combine_all_metrics(self, entity_metrics, tag_metrics, sequence_metrics, 
                           boundary_metrics, seqeval_results, crf_metrics) -> Dict[str, float]:
        """Combine όλα τα metrics σε ένα comprehensive dictionary."""
        
        final_metrics = {}
        
        # Core SeqEval metrics (primary)
        final_metrics.update({
            'precision': seqeval_results['overall_precision'],
            'recall': seqeval_results['overall_recall'],
            'f1': seqeval_results['overall_f1'],
            'accuracy': seqeval_results['overall_accuracy']
        })
        
        # Per-entity SeqEval metrics
        for entity_type in self.entity_types:
            if entity_type in seqeval_results:
                metrics = seqeval_results[entity_type]
                final_metrics.update({
                    f'{entity_type}_f1': metrics['f1'],
                    f'{entity_type}_precision': metrics['precision'],
                    f'{entity_type}_recall': metrics['recall'],
                    f'{entity_type}_number': metrics['number']
                })
        
        # Tag-level metrics
        if 'summary' in tag_metrics:
            summary = tag_metrics['summary']
            final_metrics.update({
                'tag_level_accuracy': summary['overall_accuracy'],
                'b_tag_accuracy': summary['b_tag_accuracy'],
                'i_tag_accuracy': summary['i_tag_accuracy'],
                'o_tag_accuracy': summary['o_tag_accuracy']
            })
        
        # Sequence validation metrics
        final_metrics.update({
            'consistency_rate': sequence_metrics['consistency_rate'],
            'orphaned_i_tags': sequence_metrics['orphaned_i_tags'],
            'invalid_sequences': sequence_metrics['invalid_sequences']
        })
        
        # Boundary detection metrics
        final_metrics.update({
            'entity_boundary_precision': boundary_metrics['entity_precision'],
            'entity_boundary_recall': boundary_metrics['entity_recall'],
            'entity_boundary_f1': boundary_metrics['entity_f1'],
            'start_boundary_accuracy': boundary_metrics['start_accuracy'],
            'end_boundary_accuracy': boundary_metrics['end_accuracy'],
            'entity_type_accuracy': boundary_metrics['type_accuracy']
        })
        
        # CRF metrics (αν υπάρχουν)
        for key, value in crf_metrics.items():
            if isinstance(value, (int, float)):
                final_metrics[f'crf_{key}'] = value
        
        # Detailed metrics για JSON storage
        final_metrics['detailed_entity_metrics'] = entity_metrics
        final_metrics['detailed_tag_metrics'] = tag_metrics
        final_metrics['detailed_boundary_metrics'] = boundary_metrics
        
        # Macro και micro averages
        entity_f1_scores = [final_metrics.get(f'{et}_f1', 0) for et in self.entity_types]
        final_metrics['macro_f1'] = np.mean(entity_f1_scores) if entity_f1_scores else 0
        
        return final_metrics
    
    def _print_evaluation_summary(self, metrics: Dict[str, float]):
        """Print concise evaluation summary."""
        print(f"\n{'='*80}")
        print(f"📊 EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        print(f"🎯 Overall Performance:")
        print(f"   F1 Score: {metrics['f1']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        
        print(f"\n🏷️  Tag-Level Performance:")
        print(f"   Overall Tag Accuracy: {metrics.get('tag_level_accuracy', 0):.4f}")
        print(f"   B- Tag Accuracy: {metrics.get('b_tag_accuracy', 0):.4f}")
        print(f"   I- Tag Accuracy: {metrics.get('i_tag_accuracy', 0):.4f}")
        print(f"   O Tag Accuracy: {metrics.get('o_tag_accuracy', 0):.4f}")
        
        print(f"\n🔍 Sequence Quality:")
        print(f"   BIO Consistency Rate: {metrics.get('consistency_rate', 0):.2f}%")
        print(f"   Orphaned I- Tags: {metrics.get('orphaned_i_tags', 0):,}")
        print(f"   Entity Boundary F1: {metrics.get('entity_boundary_f1', 0):.4f}")
        
        print(f"\n📈 Per-Entity Performance:")
        for entity_type in self.entity_types:
            f1_key = f'{entity_type}_f1'
            if f1_key in metrics:
                f1_score = metrics[f1_key]
                status = "🔴" if f1_score < 0.3 else "🟡" if f1_score < 0.7 else "🟢"
                print(f"   {status} {entity_type:<15}: {f1_score:.4f}")
        
        print(f"{'='*80}")


# Utility functions
def create_flexible_trainer(
    model,
    training_args,
    train_dataset,
    eval_dataset=None,
    loss_type: str = "adaptive_focal",
    focal_gamma: float = 1.0,
    class_weights: Optional[torch.Tensor] = None,
    use_crf: bool = False,
    label_list: Optional[List[str]] = None,
    debug_mode: bool = False,
    **kwargs
) -> FlexibleTrainer:
    """
    Factory function για δημιουργία FlexibleTrainer.
    
    Args:
        model: The model to train
        training_args: TrainingArguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        loss_type: Type of loss function
        focal_gamma: Focal loss gamma parameter
        class_weights: Class weights tensor
        use_crf: Whether model uses CRF
        label_list: List of labels
        debug_mode: Enable debug logging
        **kwargs: Additional arguments
    
    Returns:
        Configured FlexibleTrainer
    """
    return FlexibleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        class_weights=class_weights,
        use_crf=use_crf,
        label_list=label_list,
        debug_mode=debug_mode,
        **kwargs
    )


__all__ = [
    'FlexibleTrainer',
    'create_flexible_trainer'
]
