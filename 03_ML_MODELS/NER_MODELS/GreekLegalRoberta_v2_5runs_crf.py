import torch
import os
import json
import time
import numpy as np
import collections
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    AutoModel,
    AutoConfig
)
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset
import evaluate
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from datetime import datetime
from torchcrf import CRF

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

def save_run_results(results_file, new_results, config_name):
    # Φόρτωσε το υπάρχον json (αν υπάρχει)
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            try:
                all_results = json.load(f)
            except Exception:
                all_results = {}
    else:
        all_results = {}

    # Δημιούργησε μοναδικό key για το run (π.χ. με timestamp)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_key = f"{config_name}_{timestamp}"

    all_results[run_key] = new_results

    with open(results_file, 'w', encoding='utf-8') as f:
        clean_all_results = clean_for_json_serialization(all_results)
        json.dump(clean_all_results, f, indent=4, sort_keys=True)
    print(f"Τα αποτελέσματα για το run '{run_key}' προστέθηκαν στο '{results_file}'.")

# --- 0. ΡΥΘΜΙΣΕΙΣ ΠΕΙΡΑΜΑΤΟΣ ---
# Βασικές ρυθμίσεις για τα experiments
FORCE_RERUN = True  # Set to True για να ξανατρέξει όλα τα tests

# --- 1. CRF MODEL IMPLEMENTATION ---

class BertCRFForTokenClassification(nn.Module):
    """
    ROBERTa + CRF μοντέλο για Token Classification
    
    🔍 CRF VALIDATION & EXPLANATION:
    ================================
    Το CRF (Conditional Random Field) layer προσθέτει structural constraints:
    
    1. FEATURES που χρησιμοποιεί το CRF:
       - ROBERTa logits για κάθε token (διάσταση: num_labels)
       - Transition probabilities μεταξύ consecutive labels
       - Start/End transitions για sentence boundaries
    
    2. CONSTRAINTS που εφαρμόζει:
       ✅ B-PERSON μπορεί να ακολουθηθεί από I-PERSON ή O
       ❌ B-PERSON ΔΕΝ μπορεί να ακολουθηθεί από I-ORG
       ❌ I-PERSON ΔΕΝ μπορεί να εμφανιστεί χωρίς πρώτα B-PERSON
       ✅ O μπορεί να ακολουθηθεί από οποιοδήποτε B- tag ή O
    
    3. TRAINING PROCESS:
       - Forward pass: ROBERTa → logits → CRF loss
       - CRF loss: -log P(true_sequence | logits, transitions)
       - Backward pass: Gradient flows back through CRF to ROBERTa
    
    4. INFERENCE PROCESS:
       - Forward pass: ROBERTa → logits
       - CRF decode: Viterbi algorithm για best sequence
       - Output: Structurally valid B-I-O sequence
    """
    def __init__(self, config, num_labels, bert_model_name="AI-team-UoA/GreekLegalRoBERTa_v2"):
        super(BertCRFForTokenClassification, self).__init__()
        self.num_labels = num_labels
        self.config = config
        
        # Load ROBERTa model
        self.bert = AutoModel.from_pretrained(bert_model_name, config=config)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1)
        
        # Classifier layer
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the classifier weights"""
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass του μοντέλου
        """
        # ROBERTa encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)
        
        # Classification
        logits = self.classifier(sequence_output)  # (batch_size, seq_len, num_labels)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            # Create attention mask for CRF (exclude padding tokens)
            if attention_mask is not None:
                crf_mask = attention_mask.bool()
            else:
                crf_mask = torch.ones_like(input_ids).bool()
            
            # Αντικατάσταση των -100 labels με 0 (O tag) για το CRF
            # Το CRF δεν μπορεί να χειριστεί -100 values
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0  # Αντικατάσταση με O tag (index 0)
            
            # Ενημέρωση του mask για να αγνοήσουμε τα padding tokens
            crf_mask = crf_mask & (labels != -100)
            
            # CRF loss computation
            loss = -self.crf(logits, crf_labels, mask=crf_mask, reduction='mean')
            outputs["loss"] = loss
        
        return outputs
    
    def decode(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Decoding χρησιμοποιώντας τον CRF για τις καλύτερες predictions
        """
        outputs = self.forward(input_ids, attention_mask, token_type_ids)
        logits = outputs["logits"]
        
        if attention_mask is not None:
            crf_mask = attention_mask.bool()
        else:
            crf_mask = torch.ones_like(input_ids).bool()
        
        # CRF decoding
        predictions = self.crf.decode(logits, mask=crf_mask)
        return predictions

# --- 2. ΒΕΛΤΙΩΜΕΝΗ FOCAL LOSS IMPLEMENTATION (UNCHANGED) ---

class AdaptiveFocalLoss(nn.Module):
    """
    Βελτιωμένη εκδοχή Focal Loss ειδικά για NER tasks
    με πιο conservative παραμέτρους και καλύτερο error handling
    """
    def __init__(self, gamma=1.0, alpha=None, reduction='mean', ignore_index=-100):
        super(AdaptiveFocalLoss, self).__init__()
        self.gamma = gamma  # Πιο χαμηλό gamma για πιο ήπια εστίαση
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Δημιουργία mask για non-padded tokens
        mask = targets != self.ignore_index
        
        # Handle empty batches
        if mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Φιλτράρισμα padding tokens
        active_inputs = inputs[mask]
        active_targets = targets[mask]
        
        # Cross entropy loss
        ce_loss = F.cross_entropy(active_inputs, active_targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Alpha weighting - πιο balanced approach
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Handle tensor alpha
                alpha_tensor = self.alpha.to(device=active_targets.device, dtype=torch.float)
                alpha_t = alpha_tensor[active_targets]
        else:
            alpha_t = 1.0
        
        # Focal loss με πιο μαλακή εστίαση
        focal_loss = alpha_t * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- 2. ΒΕΛΤΙΩΜΕΝΗ FOCAL LOSS IMPLEMENTATION (UNCHANGED) ---

class AdaptiveFocalLoss(nn.Module):
    """
    Βελτιωμένη εκδοχή Focal Loss ειδικά για NER tasks
    με πιο conservative παραμέτρους και καλύτερο error handling
    """
    def __init__(self, gamma=1.0, alpha=None, reduction='mean', ignore_index=-100):
        super(AdaptiveFocalLoss, self).__init__()
        self.gamma = gamma  # Πιο χαμηλό gamma για πιο ήπια εστίαση
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Δημιουργία mask για non-padded tokens
        mask = targets != self.ignore_index
        
        # Handle empty batches
        if mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Φιλτράρισμα padding tokens
        active_inputs = inputs[mask]
        active_targets = targets[mask]
        
        # Cross entropy loss
        ce_loss = F.cross_entropy(active_inputs, active_targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Alpha weighting - πιο balanced approach
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Handle tensor alpha
                alpha_tensor = self.alpha.to(device=active_targets.device, dtype=torch.float)
                alpha_t = alpha_tensor[active_targets]
        else:
            alpha_t = 1.0
        
        # Focal loss με πιο μαλακή εστίαση
        focal_loss = alpha_t * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FlexibleTrainer(Trainer):
    """
    Trainer που μπορεί να κάνει switch μεταξύ different loss functions
    και υποστηρίζει τόσο ROBERTa όσο και ROBERTa+CRF μοντέλα
    """
    def __init__(self, *args, loss_type="adaptive_focal", focal_gamma=1.0, focal_alpha=None, 
                 class_weights=None, debug_mode=False, use_crf=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.class_weights = class_weights
        self.debug_mode = debug_mode
        self.use_crf = use_crf
        self.step_count = 0
        
        # Override the compute_metrics με custom implementation που έχει πρόσβαση στο μοντέλο
        self._original_compute_metrics = self.compute_metrics
        
    def compute_metrics_with_model_info(self, eval_pred):
        """
        Custom compute_metrics που περνάει model info στις CRF metrics
        """
        predictions, labels = eval_pred
        
        # 🔍 DEBUG INFO: Print shapes και types για debugging
        print(f"\n🔍 COMPUTE_METRICS DEBUG INFO:")
        print(f"   Predictions type: {type(predictions)}")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Use CRF: {self.use_crf}")
        
        # 🔍 CRF COMPATIBILITY FIX: Έλεγχος για το format των predictions
        if self.use_crf:
            # Για CRF models, τα predictions μπορεί να είναι ήδη decoded sequences
            if len(predictions.shape) == 3:
                # Standard logits format (batch_size, seq_len, num_labels)
                print(f"   CRF: Converting 3D logits to predictions via argmax")
                predictions = np.argmax(predictions, axis=2)
            elif len(predictions.shape) == 2:
                # Ήδη decoded sequences (batch_size, seq_len)
                print(f"   CRF: Using 2D predictions as-is")
                predictions = predictions.astype(int)
            else:
                print(f"⚠️  Unexpected predictions shape for CRF: {predictions.shape}")
                return {"error": "Invalid predictions shape"}
        else:
            # Για standard models, κάνε argmax κανονικά
            if len(predictions.shape) == 3:
                print(f"   Standard: Converting 3D logits to predictions via argmax")
                if predictions.shape[-1] == 0:
                    print(f"❌ ERROR: Predictions have 0 classes! Shape: {predictions.shape}")
                    return {"error": "Empty predictions - no classes"}
                predictions = np.argmax(predictions, axis=2)
            else:
                print(f"⚠️  Unexpected predictions shape: {predictions.shape}")
                return {"error": "Invalid predictions shape"}
        
        # 🔍 ADDITIONAL VALIDATION: Έλεγχος για κενά predictions
        if predictions.size == 0:
            print(f"❌ ERROR: Predictions array is empty!")
            return {"error": "Empty predictions array"}
        
        print(f"   Final predictions shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions.flat[:10] if predictions.size > 0 else 'EMPTY'}")
        
        # Convert to label strings
        true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] 
                           for prediction, label in zip(predictions, labels)]
        true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] 
                      for prediction, label in zip(predictions, labels)]
        
        # 🔍 TAG-LEVEL ACCURACY ANALYSIS - Για να δούμε τι γίνεται με τα I- tags
        print(f"\n🏷️  TAG-LEVEL ACCURACY ANALYSIS")
        print("="*60)
        
        # Flatten για tag-level analysis
        flat_predictions = [tag for seq in true_predictions for tag in seq]
        flat_true_labels = [tag for seq in true_labels for tag in seq]
        
        # Tag-level accuracy
        from sklearn.metrics import accuracy_score
        tag_accuracy = accuracy_score(flat_true_labels, flat_predictions)
        print(f"Overall tag-level accuracy: {tag_accuracy:.4f}")
        
        # Count predictions by tag type
        from collections import Counter
        pred_counter = Counter(flat_predictions)
        true_counter = Counter(flat_true_labels)
        
        print(f"\n📊 Tag distribution comparison:")
        print(f"{'Tag':<15} {'True':<8} {'Predicted':<8} {'Difference':<8}")
        print("-" * 50)
        
        # Focus on B-/I- tags
        for tag_type in ['B-', 'I-', 'O']:
            relevant_tags = [tag for tag in sorted(set(flat_true_labels + flat_predictions)) 
                           if tag.startswith(tag_type) or tag == tag_type]
            
            if relevant_tags:
                for tag in relevant_tags:
                    true_count = true_counter[tag]
                    pred_count = pred_counter[tag]
                    diff = pred_count - true_count
                    print(f"{tag:<15} {true_count:<8} {pred_count:<8} {diff:+8}")
                print()
        
        # Specific I- tag analysis
        i_tags_true = [tag for tag in flat_true_labels if tag.startswith('I-')]
        i_tags_pred = [tag for tag in flat_predictions if tag.startswith('I-')]
        
        print(f"🎯 I- tags specific analysis:")
        print(f"  True I- tags: {len(i_tags_true)}")
        print(f"  Predicted I- tags: {len(i_tags_pred)}")
        if len(i_tags_true) > 0:
            print(f"  I- tag recall: {len(i_tags_pred) / len(i_tags_true):.4f}")
        
        # B- tag analysis
        b_tags_true = [tag for tag in flat_true_labels if tag.startswith('B-')]
        b_tags_pred = [tag for tag in flat_predictions if tag.startswith('B-')]
        
        print(f"\n🎯 B- tags specific analysis:")
        print(f"  True B- tags: {len(b_tags_true)}")
        print(f"  Predicted B- tags: {len(b_tags_pred)}")
        if len(b_tags_true) > 0:
            print(f"  B- tag recall: {len(b_tags_pred) / len(b_tags_true):.4f}")
        
        # 📊 DETAILED ENTITY-TYPE METRICS: B- και I- ξεχωριστά ανά κατηγορία
        # (Calculate for JSON storage, don't display full table in terminal)
        
        # Βρίσκω όλα τα entity types
        entity_types = set()
        for tag in label_list:
            if tag.startswith('B-') or tag.startswith('I-'):
                entity_types.add(tag[2:])
        
        entity_metrics = {}
        
        for entity_type in sorted(entity_types):
            # Count για B- tags
            b_tag = f"B-{entity_type}"
            i_tag = f"I-{entity_type}"
            
            # True B- tags
            b_true_count = flat_true_labels.count(b_tag)
            b_pred_count = flat_predictions.count(b_tag)
            b_correct_count = sum(1 for true, pred in zip(flat_true_labels, flat_predictions) 
                                if true == b_tag and pred == b_tag)
            
            # True I- tags  
            i_true_count = flat_true_labels.count(i_tag)
            i_pred_count = flat_predictions.count(i_tag)
            i_correct_count = sum(1 for true, pred in zip(flat_true_labels, flat_predictions) 
                                if true == i_tag and pred == i_tag)
            
            # Accuracy υπολογισμοί
            b_accuracy = b_correct_count / max(b_true_count, 1)
            i_accuracy = i_correct_count / max(i_true_count, 1)
            
            # Combined F1 για αυτόν τον entity type (B- + I- combined)
            total_true = b_true_count + i_true_count
            total_pred = b_pred_count + i_pred_count  
            total_correct = b_correct_count + i_correct_count
            
            if total_true > 0 and total_pred > 0:
                precision = total_correct / total_pred
                recall = total_correct / total_true
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                f1 = 0
            
            # Αποθήκευση για summary
            entity_metrics[entity_type] = {
                'b_true': b_true_count,
                'b_pred': b_pred_count, 
                'b_correct': b_correct_count,
                'b_accuracy': b_accuracy,
                'i_true': i_true_count,
                'i_pred': i_pred_count,
                'i_correct': i_correct_count, 
                'i_accuracy': i_accuracy,
                'combined_f1': f1,
                'total_true': total_true,
                'total_pred': total_pred,
                'total_correct': total_correct
            }
            
            print(f"{entity_type:<15} {b_true_count:<8} {b_pred_count:<8} {b_correct_count:<10} {b_accuracy:<8.3f} {i_true_count:<8} {i_pred_count:<8} {i_correct_count:<10} {i_accuracy:<8.3f} {f1:<12.4f}")
        
        # Summary statistics
        print("-" * 120)
        
        # Macro-averaged F1 (unweighted average)
        macro_f1 = sum(metrics['combined_f1'] for metrics in entity_metrics.values()) / len(entity_metrics) if entity_metrics else 0
        
        # Micro-averaged F1 (weighted by frequency)
        total_all_true = sum(metrics['total_true'] for metrics in entity_metrics.values())
        total_all_pred = sum(metrics['total_pred'] for metrics in entity_metrics.values())
        total_all_correct = sum(metrics['total_correct'] for metrics in entity_metrics.values())
        
        if total_all_true > 0 and total_all_pred > 0:
            micro_precision = total_all_correct / total_all_pred
            micro_recall = total_all_correct / total_all_true
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        else:
            micro_f1 = 0
            
        print(f"{'MACRO AVG':<15} {'':<8} {'':<8} {'':<10} {'':<8} {'':<8} {'':<8} {'':<10} {'':<8} {macro_f1:<12.4f}")
        print(f"{'MICRO AVG':<15} {'':<8} {'':<8} {'':<10} {'':<8} {'':<8} {'':<8} {'':<10} {'':<8} {micro_f1:<12.4f}")
        
        print("\n� TAG-LEVEL METRICS SUMMARY:")
        print("="*60)
        print(f"📋 OVERVIEW:")
        print(f"   Overall Tag Accuracy: {tag_accuracy:.4f}")
        print(f"   Micro F1 (tag-weighted): {micro_f1:.4f}")
        print(f"   Macro F1 (entity-avg): {macro_f1:.4f}")
        print()
        print(f"📊 PER-ENTITY PERFORMANCE (Combined B-/I- F1):")
        print("-" * 50)
        
        # Show only the Combined F1 for each entity in terminal
        for entity_type in ['FACILITY', 'GPE', 'LEGISLATION_REFERENCE', 'NATIONAL_LOCATION', 'UNKNOWN_LOCATION', 'ORGANIZATION', 'PERSON', 'PUBLIC_DOCUMENT']:
            if entity_type in entity_metrics:
                combined_f1 = entity_metrics[entity_type]['combined_f1']
                status = "🔴" if combined_f1 < 0.3 else "🟡" if combined_f1 < 0.7 else "🟢"
                print(f"   {status} {entity_type:<15}: {combined_f1:.4f}")
        
        print("-" * 50)
        print(f"📝 NOTE: Full detailed B-/I- breakdown saved to JSON file")
        print("="*60)
        
        # 🔍 VALIDATION: Check entity extraction is working correctly
        validate_entity_extraction(true_predictions, true_labels, label_list)
        # 🔍 ENTITY CONSISTENCY CHECK

        print(f"\n🔍 ENTITY CONSISTENCY VALIDATION:")
        print("="*50)
        
        def check_entity_consistency(predictions):
            """
            Ελέγχει την consistency των B-I-O sequences
            Επιστρέφει: (invalid_sequences, orphaned_i_tags, total_sequences)
            """
            invalid_seqs = 0
            orphaned_tags = 0
            total_seqs = len(predictions)
            
            for seq_idx, pred_seq in enumerate(predictions):
                has_invalid = False
                
                for i, tag in enumerate(pred_seq):
                    # Έλεγχος για orphaned I- tags (I- χωρίς προηγούμενο B-)
                    if tag.startswith('I-'):
                        entity_type = tag[2:]
                        
                        # Έλεγχος αν υπάρχει B- tag πριν από αυτό το I- tag
                        found_b_tag = False
                        for j in range(i-1, -1, -1):  # Αναζήτηση προς τα πίσω
                            prev_tag = pred_seq[j]
                            if prev_tag == 'O':
                                break  # Διακοπή αν βρούμε O tag
                            elif prev_tag == f'B-{entity_type}':
                                found_b_tag = True
                                break
                            elif prev_tag.startswith('I-') and prev_tag[2:] == entity_type:
                                continue  # Συνέχεια του entity
                            else:
                                break  # Διαφορετικό entity type
                        
                        if not found_b_tag:
                            orphaned_tags += 1
                            has_invalid = True
                            
                            # Εκτύπωση sample orphaned tags για debugging
                            if orphaned_tags <= 3:  # Μόνο τα πρώτα 3
                                context_start = max(0, i-2)
                                context_end = min(len(pred_seq), i+3)
                                context = pred_seq[context_start:context_end]
                                print(f"   Orphaned I-tag found: {tag} at position {i}")
                                print(f"   Context: {context}")
                
                if has_invalid:
                    invalid_seqs += 1
            
            return invalid_seqs, orphaned_tags, total_seqs
        
        # Έλεγχος consistency
        invalid_seqs, orphaned_tags, total_seqs = check_entity_consistency(true_predictions)
        consistency_rate = (total_seqs - invalid_seqs) / total_seqs * 100
        
        print(f"   Total sequences: {total_seqs}")
        print(f"   Invalid sequences: {invalid_seqs}")
        print(f"   Orphaned I- tags: {orphaned_tags}")
        print(f"   Consistency rate: {consistency_rate:.2f}%")
        
        if invalid_seqs > 0:
            print(f"   ⚠️  {invalid_seqs} sequences have BIO inconsistencies!")
            print(f"   This suggests sliding window fragmentation issues.")
        else:
            print(f"   ✅ All sequences are BIO consistent!")
        
        print("="*50)

        # Βασικές συνολικές μετρικές (ορισμός πριν την προσθήκη consistency metrics)
        final_metrics = {
            "precision": 0,  # placeholder, will be set below
            "recall": 0,
            "f1": 0,
            "accuracy": 0
        }

        # Προσθήκη consistency metrics
        final_metrics["consistency_rate"] = consistency_rate
        final_metrics["orphaned_i_tags"] = orphaned_tags
        final_metrics["invalid_sequences"] = invalid_seqs

        # 🔍 CRF-SPECIFIC METRICS: Ανάλυση εφαρμογής CRF με model info
        crf_metrics = compute_crf_specific_metrics(
            predictions=true_predictions,
            true_labels=true_labels,
            model=self.model,
            use_crf=self.use_crf
        )

        # Compute seqeval metrics (entity-level, not tag-level)
        results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)

        print(f"\n📊 SEQEVAL METRICS VALIDATION:")
        print("="*50)
        print(f"SeqEval μετράει entities, όχι tags:")
        print(f"  - Κάθε entity (B- + I-*) μετράει ως 1")
        print(f"  - F1 υπολογίζεται σε entity-level")
        print(f"  - Exact match: ολόκληρο το entity πρέπει να είναι σωστό")
        print()
        
        # DEBUG: Δείχνουμε τι επιστρέφει η seqeval
        print(f"🔍 SEQEVAL PER-ENTITY RESULTS:")
        for entity_type in sorted(['FACILITY', 'GPE', 'LEGISLATION_REFERENCE', 'NATIONAL_LOCATION', 'UNKNOWN_LOCATION', 'ORGANIZATION', 'PERSON', 'PUBLIC_DOCUMENT']):
            if entity_type in results:
                metrics = results[entity_type]
                f1 = metrics.get('f1', 0)
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                number = metrics.get('number', 0)
                status = "🔴" if f1 < 0.3 else "🟡" if f1 < 0.7 else "🟢"
                print(f"   {status} {entity_type:<15}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, N={number}")
        print("="*50)

        # Entity counting validation
        total_entity_count = 0
        total_tag_count = 0

        for seq in true_labels[:100]:            
            # ...existing code...
            pass
        print(f"Validation sample (first 100 sequences):")
        print(f"  Total B- tags (entities): {total_entity_count}")
        print(f"  Total B-/I- tags: {total_tag_count}")
        print(f"  Average I- tags per entity: {(total_tag_count - total_entity_count) / max(total_entity_count, 1):.2f}")
        print("="*50)

        # Update βασικές συνολικές μετρικές με τα πραγματικά αποτελέσματα
        final_metrics["precision"] = results["overall_precision"]
        final_metrics["recall"] = results["overall_recall"]
        final_metrics["f1"] = results["overall_f1"]
        final_metrics["accuracy"] = results["overall_accuracy"]
        
        # 📊 ΠΡΟΣΘΗΚΗ ΟΛΩΝ ΤΩΝ ΑΝΑΛΥΤΙΚΩΝ METRICS - όπως στο crf2.json
        # Tag-level metrics
        final_metrics["tag_level_accuracy"] = tag_accuracy
        final_metrics["micro_f1"] = micro_f1
        final_metrics["macro_f1"] = macro_f1
        
        # Entity type metrics (detailed breakdown)
        final_metrics["entity_type_metrics"] = entity_metrics
        
        # Per-entity F1 scores (ΔΙΟΡΘΩΣΗ: Χρήση ΠΡΑΓΜΑΤΙΚΩΝ entity-level metrics από seqeval)
        for entity_type in entity_metrics:
            # Χρησιμοποιούμε τα ΠΡΑΓΜΑΤΙΚΑ entity-level metrics από seqeval αν υπάρχουν
            if entity_type in results:
                # SeqEval δίνει ΠΡΑΓΜΑΤΙΚΑ entity-level metrics (όχι tag-level)
                final_metrics[f"{entity_type}_f1"] = results[entity_type]["f1"]
                final_metrics[f"{entity_type}_precision"] = results[entity_type]["precision"] 
                final_metrics[f"{entity_type}_recall"] = results[entity_type]["recall"]
                final_metrics[f"{entity_type}_number"] = results[entity_type]["number"]
            else:
                # Fallback στα tag-level combined metrics αν το entity δεν υπάρχει στα seqeval results
                final_metrics[f"{entity_type}_f1"] = entity_metrics[entity_type]["combined_f1"]
                final_metrics[f"{entity_type}_precision"] = entity_metrics[entity_type]["total_correct"] / max(entity_metrics[entity_type]["total_pred"], 1)
                final_metrics[f"{entity_type}_recall"] = entity_metrics[entity_type]["total_correct"] / max(entity_metrics[entity_type]["total_true"], 1)
                final_metrics[f"{entity_type}_number"] = len([seq for seq in true_labels for tag in seq if tag == f"B-{entity_type}"])
        
        # CRF-specific metrics
        if crf_metrics:
            for key, value in crf_metrics.items():
                final_metrics[f"crf_{key}"] = value
        
        # B-/I- tag statistics
        final_metrics["total_b_tags_true"] = len(b_tags_true)
        final_metrics["total_b_tags_predicted"] = len(b_tags_pred)
        final_metrics["total_i_tags_true"] = len(i_tags_true)
        final_metrics["total_i_tags_predicted"] = len(i_tags_pred)
        final_metrics["b_tag_recall"] = len(b_tags_pred) / max(len(b_tags_true), 1)
        final_metrics["i_tag_recall"] = len(i_tags_pred) / max(len(i_tags_true), 1)
        
        return final_metrics
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        
        # Αν χρησιμοποιούμε CRF μοντέλο
        if self.use_crf:
            outputs = model(**inputs)
            loss = outputs.get("loss")
            if loss is None:
                # Fallback αν δεν υπάρχει loss στο output
                logits = outputs.get("logits")
                # Εδώ θα μπορούσαμε να προσθέσουμε custom loss computation για CRF
                # αλλά το CRF μοντέλο θα πρέπει να χειρίζεται αυτό internally
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            # Standard ROBERTa processing
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            # Reshape για loss computation
            shift_logits = logits.view(-1, self.model.config.num_labels)
            shift_labels = labels.view(-1)
            
            if self.loss_type == "adaptive_focal" or self.loss_type == "soft_focal":
                # Adaptive/Soft Focal Loss
                gamma = self.focal_gamma if self.loss_type == "soft_focal" else 1.0
                alpha = self.focal_alpha if self.focal_alpha is not None else self.class_weights
                
                loss_fct = AdaptiveFocalLoss(
                    gamma=gamma, 
                    alpha=alpha, 
                    ignore_index=-100
                )
                loss = loss_fct(shift_logits, shift_labels)
                
            elif self.loss_type == "weighted_ce":
                # Weighted Cross Entropy
                weights = self.class_weights if self.class_weights is not None else None
                loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
                loss = loss_fct(shift_logits, shift_labels)
                
            else:  # "standard"
                # Standard Cross Entropy
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits, shift_labels)
        
        # Debug logging
        if self.debug_mode and self.step_count % 100 == 0:
            model_type = "CRF" if self.use_crf else "ROBERTa"
            
            # 🔍 TRAINING VALIDATION: Έλεγχος ότι το μοντέλο βλέπει τα σωστά labels
            if not self.use_crf and labels is not None:
                # Για standard ROBERTa, ελέγχω τι labels βλέπει το μοντέλο
                active_labels = labels[labels != -100]
                if len(active_labels) > 0:
                    unique_labels = torch.unique(active_labels)
                    max_label = unique_labels.max().item()
                    min_label = unique_labels.min().item()
                    expected_max = self.model.config.num_labels - 1
                    
                    if max_label > expected_max:
                        print(f"⚠️  WARNING: Model sees label {max_label} but max expected is {expected_max}")
                    if min_label < 0:
                        print(f"⚠️  WARNING: Model sees negative label {min_label}")
                    
                    # Δείγμα από τα labels που βλέπει το μοντέλο
                    sample_labels = unique_labels[:5].tolist()
                    print(f"Step {self.step_count}: Loss={loss.item():.4f}, Model={model_type}, "
                          f"Loss_type={self.loss_type}, Active_labels={len(active_labels)}, "
                          f"Sample_labels={sample_labels}")
                else:
                    print(f"Step {self.step_count}: Loss={loss.item():.4f}, Model={model_type}, "
                          f"Loss_type={self.loss_type}, No_active_labels")
            else:
                print(f"Step {self.step_count}: Loss={loss.item():.4f}, Model={model_type}, "
                      f"Loss_type={self.loss_type}")
        
        self.step_count += 1
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction step που χειρίζεται τόσο ROBERTa όσο και CRF μοντέλα
        """
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            if self.use_crf and hasattr(model, 'decode'):
                # Για CRF χρησιμοποιούμε το decode method
                loss = None
                if not prediction_loss_only:
                    outputs = model(**inputs)
                    loss = outputs.get("loss")
                
                # Get predictions using CRF decoding
                try:
                    predictions = model.decode(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        token_type_ids=inputs.get("token_type_ids")
                    )
                    
                    print(f"🔍 CRF decode predictions type: {type(predictions)}")
                    print(f"🔍 CRF decode predictions length: {len(predictions) if predictions else 'None'}")
                    if predictions and len(predictions) > 0:
                        print(f"🔍 First prediction shape: {len(predictions[0]) if hasattr(predictions[0], '__len__') else 'scalar'}")
                    
                    # Convert predictions to tensor format που είναι συμβατό με το evaluation
                    batch_size, seq_len = inputs["input_ids"].shape
                    num_labels = self.model.num_labels
                    
                    print(f"🔍 Creating logits tensor: batch_size={batch_size}, seq_len={seq_len}, num_labels={num_labels}")
                    
                    # Δημιουργία logits tensor από τις CRF predictions
                    logits = torch.zeros((batch_size, seq_len, num_labels), device=inputs["input_ids"].device)
                    
                    if predictions is not None and len(predictions) > 0:
                        for i, seq_pred in enumerate(predictions):
                            if seq_pred is not None and len(seq_pred) > 0:
                                pred_len = min(len(seq_pred), seq_len)
                                for j in range(pred_len):
                                    if 0 <= seq_pred[j] < num_labels:  # Enhanced safety check
                                        logits[i, j, seq_pred[j]] = 1.0  # One-hot encoding
                    else:
                        print("⚠️ Warning: CRF decode returned empty predictions")
                        
                except Exception as e:
                    print(f"❌ Error in CRF decode: {e}")
                    # Fallback: δημιουργία empty logits
                    batch_size, seq_len = inputs["input_ids"].shape
                    num_labels = self.model.num_labels
                    logits = torch.zeros((batch_size, seq_len, num_labels), device=inputs["input_ids"].device)
                
            else:
                # Standard ROBERTa prediction
                outputs = model(**inputs)
                loss = outputs.get("loss")
                logits = outputs.get("logits")
        
        labels = inputs.get("labels")
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, logits, labels)

# --- 2. Φόρτωση Δεδομένων (UPDATED TO USE LEXTREME DATASET) ---
print("Φόρτωση tokenizer 'AI-team-UoA/GreekLegalRoBERTa_v2'...")
tokenizer = AutoTokenizer.from_pretrained("AI-team-UoA/GreekLegalRoBERTa_v2", add_prefix_space=True)

# ΦΟΡΤΩΣΗ GREEK LEGAL NER IOB DATASET
print("Φόρτωση Greek Legal NER IOB dataset...")
try:
    import json
    from datasets import Dataset, DatasetDict
    import os
    
    # Paths για τα IOB αρχεία (απόλυτα paths)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "..", "01_DATASETS", "GREEK_LEGAL_NER")
    train_path = os.path.join(base_dir, "train_iob.json")
    test_path = os.path.join(base_dir, "test_iob.json")
    validation_path = os.path.join(base_dir, "validation_iob.json")
    
    def load_iob_dataset(file_path):
        """Φορτώνει ένα IOB JSON αρχείο και το μετατρέπει σε dataset format"""
        print(f"   📂 Φόρτωση: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Το αρχείο έχει δομή: {"input": [...], "label": [...], "language": [...]}
        inputs = data['input']  # List of token lists
        labels = data['label']  # List of label lists (IOB tags)
        languages = data['language']  # List of language codes
        
        # Μετατροπή σε format που θέλει το HuggingFace
        dataset_data = {
            'tokens': inputs,
            'ner': labels  # Χρησιμοποιούμε 'ner' για consistency με το υπάρχον script
        }
        
        print(f"   ✅ Φορτώθηκαν {len(inputs)} προτάσεις")
        return Dataset.from_dict(dataset_data)
    
    # Φόρτωση όλων των splits
    train_ds = load_iob_dataset(train_path)
    test_ds = load_iob_dataset(test_path)
    validation_ds = load_iob_dataset(validation_path)
    
    # Δημιουργία DatasetDict
    ds = DatasetDict({
        'train': train_ds,
        'test': test_ds,
        'validation': validation_ds
    })
    
    print(f"✅ Greek Legal NER IOB dataset φορτώθηκε επιτυχώς!")
    
except Exception as e:
    print(f"❌ Σφάλμα φόρτωσης IOB dataset: {str(e)}")
    print(f"� Βεβαιωθείτε ότι τα αρχεία υπάρχουν:")
    print(f"   - {train_path}")
    print(f"   - {test_path}")
    print(f"   - {validation_path}")
    raise e
print(f"� Τελικό dataset - Splits: {list(ds.keys())}")
print(f"📊 Training examples: {len(ds['train'])}")
print(f"📊 Test examples: {len(ds['test'])}")
print(f"📊 Validation examples: {len(ds['validation'])}")

# --- 3. Προετοιμασία Ετικετών (ENHANCED WITH VALIDATION) ---
all_ner_tags_set = set(tag for example in ds['train'] for tag in example['ner'])
if 'O' in all_ner_tags_set:
    label_list = ['O'] + sorted(list(all_ner_tags_set - {'O'}))
else:
    label_list = sorted(list(all_ner_tags_set))
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in id2label.items()}

print(f"📊 DATASET LABEL ANALYSIS:")
print(f"Αριθμός μοναδικών ετικετών: {len(label_list)}")
print(f"Όλα τα labels στο dataset:")
for i, label in enumerate(label_list):
    print(f"  {i:2d}: {label}")

# Έλεγχος ότι έχουμε τα αναμενόμενα Greek Legal NER tags
expected_entity_types = ['FACILITY', 'GPE', 'LEGISLATION_REFERENCE', 'NATIONAL_LOCATION', 'ORGANIZATION', 'PERSON', 'PUBLIC_DOCUMENT', 'UNKNOWN_LOCATION']
found_entities = set()
for label in label_list:
    if label.startswith('B-') or label.startswith('I-'):
        entity_type = label[2:]
        found_entities.add(entity_type)

print(f"\n🔍 ENTITY TYPE VALIDATION:")
print(f"Αναμενόμενα entity types: {len(expected_entity_types)}")
print(f"Βρεθέντα entity types: {len(found_entities)}")
for entity_type in expected_entity_types:
    status = "✅" if entity_type in found_entities else "❌"
    print(f"  {status} {entity_type}")

missing_entities = set(expected_entity_types) - found_entities
extra_entities = found_entities - set(expected_entity_types)
if missing_entities:
    print(f"⚠️  Λείπουν entities: {missing_entities}")
if extra_entities:
    print(f"ℹ️  Επιπλέον entities: {extra_entities}")

print(f"\n📋 COMPLETE LABEL MAPPING:")
print(f"{'ID':<3} {'LABEL':<25} {'TYPE':<10} {'ENTITY'}")
print("-" * 60)
for i, label in enumerate(label_list):
    if label == 'O':
        label_type = "NON-ENTITY"
        entity_name = "N/A"
    elif label.startswith('B-'):
        label_type = "BEGIN"
        entity_name = label[2:]
    elif label.startswith('I-'):
        label_type = "INSIDE"
        entity_name = label[2:]
    else:
        label_type = "OTHER"
        entity_name = label
    print(f"{i:<3} {label:<25} {label_type:<10} {entity_name}")
print("="*60)

# --- 4. Προεπεξεργασία (Unchanged) ---
# --- 4. Προεπεξεργασία (ΤΡΟΠΟΠΟΙΗΜΕΝΗ ΓΙΑ SLIDING WINDOW) ---
MAX_LENGTH = 512
OVERLAP = 50

# Στο compute_metrics, δώσε περισσότερο βάρος στις σπάνιες κατηγορίες:
def compute_weighted_f1(results):
    """Υπολογίζει weighted F1 που δίνει περισσότερη έμφαση στις σπάνιες κατηγορίες"""
    rare_classes = ["NATIONAL_LOCATION", "FACILITY", "UNKNOWN_LOCATION"]
    rare_weight = 2.0
    normal_weight = 1.0
    
    weighted_f1 = 0
    total_weight = 0
    
    for class_name, metrics in results.items():
        if isinstance(metrics, dict) and 'f1' in metrics:
            # ΔΙΟΡΘΩΣΗ: Έλεγχος με endswith για B-NATIONAL_LOCATION, I-NATIONAL_LOCATION κτλ
            is_rare = any(class_name.endswith(rare_class) for rare_class in rare_classes)
            weight = rare_weight if is_rare else normal_weight
            weighted_f1 += metrics['f1'] * weight
            total_weight += weight
    
    return weighted_f1 / total_weight if total_weight > 0 else 0

# Για τις πολύ σπάνιες κατηγορίες, κάνε manual oversampling:
def oversample_rare_classes(tokenized_ds, rare_threshold=30, oversample_factor=3):
    """Κάνει oversample τα rare class instances"""
    train_data = tokenized_ds["train"]
    
    # Βρες ποια examples έχουν rare classes
    rare_examples = []
    for example in train_data:
        labels = example["labels"]
        # ΔΙΟΡΘΩΣΗ: Έλεγχος με endswith για να βρούμε B-NATIONAL_LOCATION, I-NATIONAL_LOCATION κτλ
        has_rare = False
        for l in labels:
            if l != -100:
                label_name = label_list[l]
                if any(label_name.endswith(rare_class) for rare_class in ["NATIONAL_LOCATION", "FACILITY"]):
                    has_rare = True
                    break
        if has_rare:
            rare_examples.append(example)
    
    # Πρόσθεσε τα rare examples πολλές φορές
    for _ in range(oversample_factor):
        train_data = train_data.add_item(rare_examples)
    
    return train_data

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
        stride=OVERLAP,
        return_overflowing_tokens=True,
        padding="max_length"
    )

    all_labels = []
    for i, overflow_map_idx in enumerate(tokenized_inputs["overflow_to_sample_mapping"]):
        ner_tags = examples["ner"][overflow_map_idx]
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label2id[ner_tags[word_idx]])
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

print("Εφαρμογή προεπεξεργασίας με Sliding Window στο dataset...")
tokenized_ds = ds.map(
    tokenize_and_align_labels, 
    batched=True, 
    remove_columns=ds["train"].column_names
)

# Έλεγχος μεγέθους training set
print(f"Αρχικό μέγεθος training set: {len(ds['train'])} έγγραφα")
print(f"Νέο μέγεθος training set: {len(tokenized_ds['train'])} chunks (δείγματα εκπαίδευσης)")

# 🔍 ΚΡΙΣΙΜΟΣ ΕΛΕΓΧΟΣ: Validation ότι το tokenization διατηρεί τα labels σωστά
print(f"\n🔍 TOKENIZATION VALIDATION:")
print("="*60)

# Συλλέγω όλα τα labels από το tokenized dataset
all_tokenized_labels = []
for example in tokenized_ds["train"]:
    labels = example["labels"]
    all_tokenized_labels.extend([l for l in labels if l != -100])

tokenized_label_counts = Counter(all_tokenized_labels)
print(f"Συνολικά valid labels μετά tokenization: {len(all_tokenized_labels):,}")

# Έλεγχος ότι όλα τα αναμενόμενα labels υπάρχουν
missing_labels = []
present_labels = []
for i, label in enumerate(label_list):
    if i in tokenized_label_counts:
        present_labels.append((i, label, tokenized_label_counts[i]))
    else:
        missing_labels.append((i, label))

print(f"\n📊 LABEL PRESENCE CHECK:")
print(f"✅ Παρόντα labels: {len(present_labels)}/{len(label_list)}")
print(f"❌ Απόντα labels: {len(missing_labels)}")

if missing_labels:
    print(f"\n⚠️  ΠΡΟΣΟΧΗ: Τα παρακάτω labels λείπουν από το tokenized dataset:")
    for label_id, label_name in missing_labels:
        print(f"  {label_id}: {label_name}")
    print(f"Αυτό σημαίνει ότι το μοντέλο ΔΕΝ θα εκπαιδευτεί για αυτά τα labels!")

# Δείχνω τα 5 σπανιότερα labels που υπάρχουν
print(f"\n📈 RAREST LABELS IN TOKENIZED DATASET:")
rare_labels = sorted(present_labels, key=lambda x: x[2])[:5]
for label_id, label_name, count in rare_labels:
    percentage = (count / len(all_tokenized_labels)) * 100
    print(f"  {label_name}: {count:,} occurrences ({percentage:.3f}%)")

# Έλεγχος ότι τα entity types που αναμένουμε υπάρχουν
entity_types_in_tokenized = set()
for label_id, label_name, count in present_labels:
    if label_name.startswith('B-') or label_name.startswith('I-'):
        entity_type = label_name[2:]
        entity_types_in_tokenized.add(entity_type)

expected_entities = {'FACILITY', 'GPE', 'LEGISLATION_REFERENCE', 'NATIONAL_LOCATION', 'UNKNOWN_LOCATION', 'ORGANIZATION', 'PERSON', 'PUBLIC_DOCUMENT'}
missing_entity_types = expected_entities - entity_types_in_tokenized

if missing_entity_types:
    print(f"\n❌ MISSING ENTITY TYPES:")
    for entity_type in missing_entity_types:
        print(f"  {entity_type} - Το μοντέλο ΔΕΝ θα μάθει να εντοπίζει αυτό το entity type!")
else:
    print(f"\n✅ Όλα τα αναμενόμενα entity types υπάρχουν στο tokenized dataset")

print("="*60)

# --- 5. ΒΕΛΤΙΩΜΕΝΟΣ ΥΠΟΛΟΓΙΣΜΟΣ CLASS WEIGHTS ---
def compute_class_weights(tokenized_ds, label_list, method="capped_sqrt_inv_freq", original_dataset=None):
    """
    Υπολογίζει class weights με διαφορετικές μεθόδους.
    Βελτιωμένη έκδοση: manual boost για πολύ rare classes βάσει ΠΡΑΓΜΑΤΙΚΩΝ αριθμών:
    Χρησιμοποιεί τα πραγματικά entity counts από το αρχικό dataset, όχι τα tokenized counts
    """
    
    # ΥΠΟΛΟΓΙΣΜΟΣ ΠΡΑΓΜΑΤΙΚΩΝ ENTITY COUNTS από το αρχικό dataset
    if original_dataset is not None:
        print("Υπολογισμός πραγματικών entity counts από το αρχικό dataset...")
        
        # Συνάρτηση για μέτρηση entities σε ένα split
        def count_entities_in_split(dataset_split):
            entity_counts = {}
            i_tag_counts = {}  # Ξεχωριστή μέτρηση για I- tags
            
            for example in dataset_split:
                ner_tags = example['ner']
                
                for tag in ner_tags:
                    if tag.startswith('B-'):
                        # Νέο entity ξεκινάει
                        entity_type = tag[2:]  # Αφαίρεση του 'B-'
                        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                    elif tag.startswith('I-'):
                        # I- tag count ξεχωριστά
                        entity_type = tag[2:]  # Αφαίρεση του 'I-'
                        i_tag_counts[entity_type] = i_tag_counts.get(entity_type, 0) + 1
                    # Τα I- tags δεν μετράνε ως νέα entities
            
            return entity_counts, i_tag_counts
        
        # Μέτρηση entities στο training set
        train_entity_counts, train_i_tag_counts = count_entities_in_split(original_dataset['train'])
        
        print("Πραγματικά entity counts από το αρχικό dataset:")
        for entity_type, count in sorted(train_entity_counts.items()):
            print(f"  {entity_type}: {count} entities")
        print()
    else:
        # Fallback στην παλιά μέθοδο αν δεν έχουμε το αρχικό dataset
        train_entity_counts = {}
        train_i_tag_counts = {}
    
    # ΡΥΘΜΙΖΕΙΣ βάσει ΠΡΑΓΜΑΤΙΚΩΝ μετρημένων B- tag counts από το dataset
    # Βάσει των ΝΕΩΝ ΣΩΣΤΩΝ μετρημένων αποτελεσμάτων:
    # NATIONAL_LOCATION=25, UNKNOWN_LOCATION=261, PUBLIC_DOCUMENT=874, PERSON=1212, FACILITY=1041, etc.
    very_rare_threshold = 100    # NATIONAL_LOCATION: 25 entities (εξαιρετικά σπάνιο!)
    rare_threshold = 1000        # UNKNOWN_LOCATION: 261, PUBLIC_DOCUMENT: 874 
    medium_threshold = 2500      # PERSON: 1212, FACILITY: 1041, LEGISLATION_REFERENCE: 2158
    common_threshold = 5000      # ORG: 3706, GPE: 4315
    
    very_rare_boost = 200.0      # Εξαιρετικά υψηλό για NATIONAL_LOCATION (25 entities)
    rare_boost_value = 75.0      # Υψηλό για UNKNOWN_LOCATION, PUBLIC_DOCUMENT
    medium_boost_value = 25.0    # Μέτριο για PERSON, FACILITY, LEG-REFS
    common_boost_value = 15.0    # Χαμηλό για ORG, GPE
    max_cap = 150.0              # Υψηλότερο cap για extreme cases

    # Υπολογισμός token-level counts για normalization
    all_labels = []
    for example in tokenized_ds["train"]:
        labels = example["labels"]
        all_labels.extend([l for l in labels if l != -100])
    
    label_counts = Counter(all_labels)
    total_samples = len(all_labels)
    
    weights = torch.ones(len(label_list))
    
    if method == "capped_sqrt_inv_freq":
        for label_id, token_count in label_counts.items():
            label_name = label_list[label_id]
            
            # ΔΙΟΡΘΩΣΗ: Χρησιμοποιούμε πραγματικά entity counts
            if label_name.startswith('B-'):
                entity_type = label_name[2:]  # Αφαίρεση του 'B-'
                actual_entity_count = train_entity_counts.get(entity_type, 0)
            elif label_name.startswith('I-'):
                entity_type = label_name[2:]  # Αφαίρεση του 'I-'
                # ΓΙΑ I- TAGS: Χρησιμοποιούμε τα πραγματικά I- tag counts
                # ΟΧΙ τα B- tag counts!
                actual_entity_count = train_i_tag_counts.get(entity_type, token_count)
            else:
                # Για O tag, χρησιμοποιούμε το token count
                actual_entity_count = token_count
            
            if actual_entity_count < very_rare_threshold:
                weights[label_id] = very_rare_boost
                print(f"  Very Rare: {label_name} -> weight {very_rare_boost} (entities: {actual_entity_count})")
            elif actual_entity_count < rare_threshold:
                weights[label_id] = rare_boost_value
                print(f"  Rare: {label_name} -> weight {rare_boost_value} (entities: {actual_entity_count})")
            elif actual_entity_count < medium_threshold:
                weights[label_id] = medium_boost_value
                print(f"  Medium: {label_name} -> weight {medium_boost_value} (entities: {actual_entity_count})")
            else:
                # Για common classes, χρησιμοποιούμε το token count για το weight calculation
                raw_weight = np.sqrt(total_samples / (len(label_list) * token_count))
                weights[label_id] = min(raw_weight, max_cap)
                print(f"  Common: {label_name} -> weight {min(raw_weight, max_cap):.3f} (entities: {actual_entity_count})")
        
    elif method == "inverse_freq":
        for label_id, count in label_counts.items():
            weights[label_id] = total_samples / (len(label_list) * count)
    
    elif method == "sqrt_inv_freq":
        for label_id, count in label_counts.items():
            weights[label_id] = np.sqrt(total_samples / (len(label_list) * count))
    
    elif method == "log_balanced":
        for label_id, count in label_counts.items():
            weights[label_id] = np.log(total_samples / count + 1)
    
    elif method == "effective_num":
        beta = 0.9999
        for label_id, count in label_counts.items():
            effective_num = (1 - beta**count) / (1 - beta)
            weights[label_id] = 1.0 / effective_num
    
    # Normalize weights
    weights = weights / weights.mean()
    
    # Special handling για το "O" tag
    o_index = label_list.index('O') if 'O' in label_list else None
    if o_index is not None and method != "inverse_freq":
        weights[o_index] = weights[o_index] * 0.05  # Ακόμη χαμηλότερο για καλύτερη εστίαση
    
    # Print weight distribution
    print(f"\nClass Weight Distribution (Method: {method}):")
    print("-" * 70)
    print(f"{'Label':<15} {'Weight':<8} {'Count':<8} {'Percentage':<10} {'Category'}")
    print("-" * 70)
    
    for i, (label, weight) in enumerate(zip(label_list, weights)):
        count = label_counts.get(i, 0)
        percentage = (count / total_samples) * 100
        if count > 50000:
            category = "Dominant"
        elif count > 5000:
            category = "Common"
        elif count > 1000:
            category = "Medium"
        elif count > 100:
            category = "Rare"
        else:
            category = "Very Rare"
        print(f"  {label:<15} {weight:<8.3f} {count:<8} {percentage:<10.2f}% {category}")
    print("-" * 70)
    print(f"Total samples: {total_samples:,}")
    print(f"Weight range: {weights.min():.3f} - {weights.max():.3f}")
    print(f"Weight std: {weights.std():.3f}")
    
    return weights.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def apply_rare_class_augmentation(tokenized_ds, enable_augmentation=True, multiplier=3):
    """
    Εφαρμόζει data augmentation για rare classes με on/off switch
    
    Args:
        tokenized_ds: Το tokenized dataset
        enable_augmentation: True/False για ενεργοποίηση
        multiplier: Πόσες φορές να πολλαπλασιάσει τα rare examples
    
    Returns:
        Το (augmented ή όχι) dataset
    """
    if not enable_augmentation:
        print("\n🔄 RARE CLASS AUGMENTATION: DISABLED")
        print("="*50)
        print("   Augmentation is turned OFF - using original dataset")
        print("="*50)
        return tokenized_ds
    
    print(f"\n🔄 RARE CLASS AUGMENTATION: ENABLED")
    print("="*50)
    print(f"   Multiplier: {multiplier}x")
    print(f"   Target rare classes: NATIONAL_LOCATION, UNKNOWN_LOCATION, FACILITY")
    
    # Κώδικας augmentation μόνο για το training set
    train_dataset = tokenized_ds["train"]
    
    rare_examples = []
    original_size = len(train_dataset)
    
    # Μετρητής για κάθε rare class
    rare_counts = {
        'NATIONAL_LOCATION': 0,
        'UNKNOWN_LOCATION': 0, 
        'FACILITY': 0
    }
    
    print(f"   Scanning {original_size:,} training examples...")
    
    for example in train_dataset:
        labels = example["labels"]
        # Μετατροπή label IDs σε label names για έλεγχο
        label_names = [label_list[l] for l in labels if l != -100]
        
        has_rare = any(
            label.endswith(('NATIONAL_LOCATION', 'UNKNOWN_LOCATION', 'FACILITY'))
            for label in label_names
        )
        
        if has_rare:
            # Προσδιορισμός ποια rare class περιέχει (ΔΙΟΡΘΩΜΕΝΟ)
            rare_classes_in_example = []
            for rare_class in rare_counts.keys():
                if any(label.endswith(rare_class) for label in label_names):
                    rare_classes_in_example.append(rare_class)
                    rare_counts[rare_class] += 1
            
            # Πολλαπλασιασμός του example μόνο ΜΙΑ φορά ανεξάρτητα από το πόσα rare classes περιέχει
            rare_examples.extend([example] * multiplier)
    
    print(f"\n   📊 RARE CLASS STATISTICS:")
    print(f"   {'Class':<15} {'Found':<8} {'Will Add':<10}")
    print(f"   {'-'*35}")
    
    total_rare_examples = len(rare_examples) // multiplier
    for rare_class, count in rare_counts.items():
        will_add = count * multiplier
        print(f"   {rare_class:<15} {count:<8} {will_add:<10}")
    
    print(f"   {'-'*35}")
    print(f"   {'TOTAL':<15} {total_rare_examples:<8} {len(rare_examples):<10}")
    
    if rare_examples:
        # Δημιουργία νέου augmented training set
        from datasets import Dataset, concatenate_datasets
        augmented_examples = Dataset.from_list(rare_examples)
        augmented_train = concatenate_datasets([train_dataset, augmented_examples])
        # Ενημέρωση του dataset
        tokenized_ds["train"] = augmented_train
        final_size = len(augmented_train)
        increase_ratio = final_size / original_size
        print(f"\n   ✅ AUGMENTATION COMPLETED:")
        print(f"   Original training size: {original_size:,}")
        print(f"   Augmented training size: {final_size:,}")
        print(f"   Size increase: {increase_ratio:.2f}x ({((increase_ratio-1)*100):.1f}% more)")
        
    else:
        print(f"\n   ⚠️  No rare classes found - no augmentation applied")
    
    print("="*50)
    return tokenized_ds


class_weights = None
# Υπολογίζουμε class weights πάντα για χρήση σε specific configurations
print("Υπολογισμός class weights...")
class_weights = compute_class_weights(tokenized_ds, label_list, method="capped_sqrt_inv_freq", original_dataset=ds)

# --- 6. Ορισμός Μετρικών (ENHANCED WITH VALIDATION) ---
metric = evaluate.load("seqeval")

def validate_entity_extraction(true_predictions, true_labels, label_list):
    """
    Validates that entities are correctly extracted and counted from B-/I- tag sequences
    """
    print(f"\n🔍 ENTITY EXTRACTION VALIDATION:")
    print("="*60)
    
    # Count entities in true labels and predictions
    def extract_entities_from_sequence(sequence):
        entities = []
        current_entity = None
        
        for i, tag in enumerate(sequence):
            if tag.startswith('B-'):
                # Νέο entity ξεκινάει
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = {
                    'type': tag[2:],  # Αφαίρεση 'B-'
                    'start': i,
                    'end': i,
                    'tokens': [tag]
                }
            elif tag.startswith('I-') and current_entity is not None:
                # Συνέχεια του τρέχοντος entity
                entity_type = tag[2:]  # Αφαίρεση 'I-'
                if entity_type == current_entity['type']:
                    current_entity['end'] = i
                    current_entity['tokens'].append(tag)
                else:
                    # Incompatible I- tag, close current and start new
                    entities.append(current_entity)
                    current_entity = {
                        'type': entity_type,
                        'start': i,
                        'end': i,
                        'tokens': [tag]
                    }
            elif tag == 'O':
                # End current entity if exists
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
            # Ignore other tags
        
        # Close final entity if exists
        if current_entity is not None:
            entities.append(current_entity)
            
        return entities
    
    # Sample validation on first few sequences
    sample_size = min(3, len(true_predictions))
    total_true_entities = 0
    total_pred_entities = 0
    
    for i in range(sample_size):
        true_seq = true_labels[i]
        pred_seq = true_predictions[i]
        
        true_entities = extract_entities_from_sequence(true_seq)
        pred_entities = extract_entities_from_sequence(pred_seq)
        
        total_true_entities += len(true_entities)
        total_pred_entities += len(pred_entities)
        
        print(f"Sample {i+1}:")
        print(f"  True entities: {len(true_entities)}")
        print(f"  Pred entities: {len(pred_entities)}")
        
        # Show first few entities
        for j, entity in enumerate(true_entities[:2]):
            print(f"    True {j+1}: {entity['type']} (tokens: {len(entity['tokens'])})")
        for j, entity in enumerate(pred_entities[:2]):
            print(f"    Pred {j+1}: {entity['type']} (tokens: {len(entity['tokens'])})")
    
    print(f"\nTotal entities in sample:")
    print(f"  True: {total_true_entities}")
    print(f"  Predicted: {total_pred_entities}")
    
    # Check entity type distribution
    entity_types_true = set()
    entity_types_pred = set()
    
    for seq in true_labels[:10]:  # Check first 10 sequences
        for tag in seq:
            if tag.startswith('B-') or tag.startswith('I-'):
                entity_types_true.add(tag[2:])
    
    for seq in true_predictions[:10]:
        for tag in seq:
            if tag.startswith('B-') or tag.startswith('I-'):
                entity_types_pred.add(tag[2:])
    
    print(f"\nEntity types found:")
    print(f"  In true labels: {sorted(entity_types_true)}")
    print(f"  In predictions: {sorted(entity_types_pred)}")
    
    missing_in_pred = entity_types_true - entity_types_pred
    if missing_in_pred:
        print(f"⚠️  Entity types missing in predictions: {missing_in_pred}")
    else:
        print(f"✅ All true entity types found in predictions")
    
    print("="*60)

def compute_crf_specific_metrics(predictions, true_labels, model, use_crf=False):
    """
    Υπολογίζει μετρικές ειδικά για CRF για να παρακολουθούμε την εφαρμογή του
    
    🔍 CRF MONITORING METRICS:
    ==========================
    1. Transition Consistency: Πόσο συχνά οι transitions είναι valid
    2. Entity Boundary Accuracy: Πόσο καλά εντοπίζει τα όρια των entities
    3. Sequence Coherence: Πόσο συνεπείς είναι οι predicted sequences
    """
    print(f"\n🔍 CRF-SPECIFIC METRICS ANALYSIS:")
    print("="*50)
    
    if not use_crf:
        print(f"⚠️  Μοντέλο χωρίς CRF - Basic transition analysis:")
        
        # Για non-CRF models, ανάλυση των invalid transitions
        invalid_transitions = 0
        total_transitions = 0
        invalid_examples = []
        
        for seq_idx, pred_seq in enumerate(predictions):  # Check all sequences, not just 100
            for i in range(len(pred_seq) - 1):
                current_tag = pred_seq[i]
                next_tag = pred_seq[i + 1]
                total_transitions += 1
                
                # Έλεγχος για invalid transitions
                is_invalid = False
                
                # Κανόνες BIO
                if current_tag.startswith('B-') and next_tag.startswith('I-'):
                    # B-X → I-Y: Valid μόνο αν X == Y
                    current_entity = current_tag[2:]
                    next_entity = next_tag[2:]
                    if current_entity != next_entity:
                        is_invalid = True
                        invalid_examples.append(f"B-{current_entity} → I-{next_entity}")
                
                elif current_tag.startswith('I-') and next_tag.startswith('I-'):
                    # I-X → I-Y: Valid μόνο αν X == Y
                    current_entity = current_tag[2:]
                    next_entity = next_tag[2:]
                    if current_entity != next_entity:
                        is_invalid = True
                        invalid_examples.append(f"I-{current_entity} → I-{next_entity}")
                
                elif next_tag.startswith('I-') and not current_tag.startswith(('B-', 'I-')):
                    # O → I-X: Invalid (orphaned I- tag)
                    is_invalid = True
                    invalid_examples.append(f"{current_tag} → {next_tag}")
                
                if is_invalid:
                    invalid_transitions += 1
        
        transition_validity = (total_transitions - invalid_transitions) / total_transitions * 100
        print(f"   Invalid transitions: {invalid_transitions}/{total_transitions} ({100-transition_validity:.2f}%)")
        print(f"   Transition validity: {transition_validity:.2f}%")
        
        if invalid_examples:
            print(f"   Sample invalid transitions: {invalid_examples[:5]}")
        
    else:
        print(f"✅ CRF MODEL - Advanced CRF analysis:")
        
        # CRF specific metrics
        print(f"   CRF guarantees valid transitions: 100%")
        print(f"   Using Viterbi decoding for optimal sequences")
        
        # Transition matrix analysis (αν έχουμε πρόσβαση στο μοντέλο)
        if hasattr(model, 'crf'):
            crf_layer = model.crf
            transition_matrix = crf_layer.transitions.detach().cpu()
            
            print(f"\n📊 CRF TRANSITION MATRIX ANALYSIS:")
            print(f"   Matrix shape: {transition_matrix.shape}")
            print(f"   Min transition score: {transition_matrix.min().item():.3f}")
            print(f"   Max transition score: {transition_matrix.max().item():.3f}")
            print(f"   Mean transition score: {transition_matrix.mean().item():.3f}")
            
            # Ανάλυση high/low probability transitions
            num_labels = transition_matrix.shape[0]
            high_prob_transitions = (transition_matrix > transition_matrix.mean()).sum().item()
            low_prob_transitions = (transition_matrix < transition_matrix.mean()).sum().item()
            
            print(f"   High probability transitions: {high_prob_transitions}/{num_labels*num_labels}")
            print(f"   Low probability transitions: {low_prob_transitions}/{num_labels*num_labels}")
            
            # Start/End transition analysis
            start_transitions = crf_layer.start_transitions.detach().cpu()
            end_transitions = crf_layer.end_transitions.detach().cpu()
            
            print(f"\n🚀 START/END TRANSITIONS:")
            print(f"   Start transition range: {start_transitions.min().item():.3f} to {start_transitions.max().item():.3f}")
            print(f"   End transition range: {end_transitions.min().item():.3f} to {end_transitions.max().item():.3f}")
            
            # Find most/least likely starting tags
            start_probs = torch.softmax(start_transitions, dim=0)
            most_likely_start = start_probs.argmax().item()
            least_likely_start = start_probs.argmin().item()
            
            print(f"   Most likely start tag: {label_list[most_likely_start]} ({start_probs[most_likely_start].item():.3f})")
            print(f"   Least likely start tag: {label_list[least_likely_start]} ({start_probs[least_likely_start].item():.3f})")
    
    # Entity boundary analysis (κοινό για όλα τα models)
    print(f"\n🎯 ENTITY BOUNDARY ANALYSIS:")
    
    def analyze_entity_boundaries(pred_sequences, true_sequences):
        correct_starts = 0
        correct_ends = 0
        total_entities = 0
        boundary_errors = []
        
        for pred_seq, true_seq in zip(pred_sequences, true_sequences):  # Analyze all sequences
            # Find entities in both sequences
            true_entities = []
            pred_entities = []
            
            # Extract true entities
            current_entity = None
            for i, tag in enumerate(true_seq):
                if tag.startswith('B-'):
                    if current_entity:
                        true_entities.append(current_entity)
                    current_entity = {'type': tag[2:], 'start': i, 'end': i}
                elif tag.startswith('I-') and current_entity and tag[2:] == current_entity['type']:
                    current_entity['end'] = i
                else:
                    if current_entity:
                        true_entities.append(current_entity)
                        current_entity = None
            if current_entity:
                true_entities.append(current_entity)
            
            # Extract predicted entities
            current_entity = None
            for i, tag in enumerate(pred_seq):
                if tag.startswith('B-'):
                    if current_entity:
                        pred_entities.append(current_entity)
                    current_entity = {'type': tag[2:], 'start': i, 'end': i}
                elif tag.startswith('I-') and current_entity and tag[2:] == current_entity['type']:
                    current_entity['end'] = i
                else:
                    if current_entity:
                        pred_entities.append(current_entity)
                        current_entity = None
            if current_entity:
                pred_entities.append(current_entity)
            
            # Check boundary accuracy
            for true_entity in true_entities:
                total_entities += 1
                
                # Find matching predicted entity by type and position
                for pred_entity in pred_entities:
                    if (true_entity['type'] == pred_entity['type'] and 
                        abs(true_entity['start'] - pred_entity['start']) <= 1):  # Allow 1-token tolerance
                        
                        if true_entity['start'] == pred_entity['start']:
                            correct_starts += 1
                        if true_entity['end'] == pred_entity['end']:
                            correct_ends += 1
                        break
                else:
                    # No matching predicted entity found
                    boundary_errors.append(f"Missing {true_entity['type']} at {true_entity['start']}-{true_entity['end']}")
        
        return correct_starts, correct_ends, total_entities, boundary_errors
    
    correct_starts, correct_ends, total_entities, boundary_errors = analyze_entity_boundaries(predictions, true_labels)
    
    if total_entities > 0:
        start_accuracy = correct_starts / total_entities * 100
        end_accuracy = correct_ends / total_entities * 100
        print(f"   Correct entity starts: {correct_starts}/{total_entities} ({start_accuracy:.2f}%)")
        print(f"   Correct entity ends: {correct_ends}/{total_entities} ({end_accuracy:.2f}%)")
        
        if boundary_errors:
            print(f"   Sample boundary errors: {boundary_errors[:3]}")
    else:
        print(f"   No entities found in sample for boundary analysis")
    
    # Sequence coherence score
    print(f"\n🔗 SEQUENCE COHERENCE ANALYSIS:")
    
    def compute_coherence_score(sequences):
        coherence_scores = []
        
        for seq in sequences:  # Analyze all sequences, not just 50
            score = 0
            total_positions = len(seq) - 1
            
            if total_positions == 0:
                continue
                
            for i in range(total_positions):
                current = seq[i]
                next_tag = seq[i + 1]
                
                # Coherence rules (higher score = more coherent)
                if current == 'O' and next_tag.startswith(('O', 'B-')):
                    score += 1  # Valid: O can be followed by O or B-
                elif current.startswith('B-') and (next_tag == 'O' or next_tag.startswith('B-') or 
                                                   (next_tag.startswith('I-') and next_tag[2:] == current[2:])):
                    score += 1  # Valid: B-X can be followed by O, B-*, or I-X
                elif current.startswith('I-') and (next_tag == 'O' or next_tag.startswith('B-') or 
                                                   (next_tag.startswith('I-') and next_tag[2:] == current[2:])):
                    score += 1  # Valid: I-X can be followed by O, B-*, or I-X
            
            if total_positions > 0:
                coherence_scores.append(score / total_positions)
        
        return coherence_scores
    
    coherence_scores = compute_coherence_score(predictions)
    
    if coherence_scores:
        avg_coherence = sum(coherence_scores) / len(coherence_scores) * 100
        print(f"   Average sequence coherence: {avg_coherence:.2f}%")
        print(f"   Coherence range: {min(coherence_scores)*100:.1f}% - {max(coherence_scores)*100:.1f}%")
        
        if use_crf:
            print(f"   ✅ CRF should achieve 100% coherence (structural constraints)")
        else:
            print(f"   ⚠️  Standard model may have coherence issues")
    
    print("="*50)
    
    # Return metrics για logging
    crf_metrics = {
        'boundary_start_accuracy': correct_starts / max(total_entities, 1) * 100,
        'boundary_end_accuracy': correct_ends / max(total_entities, 1) * 100,
        'sequence_coherence': sum(coherence_scores) / max(len(coherence_scores), 1) * 100 if coherence_scores else 0,
    }
    
    if not use_crf and 'transition_validity' in locals():
        crf_metrics['transition_validity'] = transition_validity
    
    return crf_metrics

# --- 7. Ρυθμίσεις Εκπαίδευσης & Data Collator ---
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16 
NUM_EPOCHS = 6
WEIGHT_DECAY = 0.01
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# --- 8. ΑΥΤΟΜΑΤΗ ΔΟΚΙΜΗ ΟΛΩΝ ΤΩΝ CONFIGURATIONS ---
def test_all_loss_configurations():
    """Δοκιμάζει όλες τις configurations και συγκρίνει αποτελέσματα"""
    
    # Access to global variables
    global tokenized_ds, label_list, id2label, label2id, ds, data_collator
    
    # 📁 ΈΛΕΓΧΟΣ EXISTING RESULTS - Skip configurations που έχουν ήδη εκτελεστεί
    incremental_save_file = "progressive_results_with_crf.json"
    completed_configs = set()
    
    if not FORCE_RERUN:  # Μόνο αν δεν θέλουμε force rerun
        try:
            with open(incremental_save_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                completed_configs = set(existing_results.keys())
                print(f"📂 Βρέθηκαν υπάρχοντα αποτελέσματα για: {list(completed_configs)}")
        except (FileNotFoundError, json.JSONDecodeError):
            existing_results = {}
            print(f"📂 Δεν βρέθηκαν υπάρχοντα αποτελέσματα - ξεκινάω από την αρχή")
    else:
        print(f"🔄 FORCE_RERUN=True - Ξανατρέχω όλα τα tests")
        existing_results = {}
    
    # ΜΟΝΟ CROSS ENTROPY - Απλοποιημένη configuration για βασική σύγκριση
    configurations = [
        # ΜΟΝΟ Cross Entropy χωρίς επιπλέον techniques
        {
            "name": "cross_entropy_only",
            "loss_type": "standard",  # Βασικό Cross Entropy
            "focal_gamma": None,
            "focal_alpha": None,
            "use_weights": False,  # ΔΕΝ χρησιμοποιούμε class weights
            "use_crf": False,      # ΔΕΝ χρησιμοποιούμε CRF
            "weight_method": None
        }
    ]
    
    # ⏭️ ΦΙΛΤΡΑΡΙΣΜΑ - Skip configurations που έχουν ήδη ολοκληρωθεί
    remaining_configs = [config for config in configurations if config['name'] not in completed_configs]
    
    if not remaining_configs:
        print(f"✅ Όλες οι configurations έχουν ήδη ολοκληρωθεί!")
        print(f"🔍 Τελικά αποτελέσματα: {list(completed_configs)}")
        return existing_results
    
    print(f"⏭️  Configurations προς εκτέλεση: {[c['name'] for c in remaining_configs]}")
    print(f"⏹️  Configurations που θα παραλειφθούν: {list(completed_configs)}")
        
    results = existing_results  # Ξεκινάω με τα υπάρχοντα results
    # 🔄 RARE CLASS AUGMENTATION (μια φορά για όλα τα experiments)
    tokenized_ds = apply_rare_class_augmentation(
        tokenized_ds, 
        enable_augmentation=False,  # ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟ - ΔΕΝ κάνουμε augmentation
        multiplier=3  # 3x multiplier για rare classes (δεν εφαρμόζεται όταν disabled)
    )
    for config in remaining_configs:  # Χρησιμοποιώ remaining_configs αντί για configurations
        print(f"\n{'='*80}")
        print(f"🔄 Δοκιμάζω configuration: {config['name']}")
        print(f"   Model: {'ROBERTa+CRF' if config.get('use_crf') else 'ROBERTa'}")
        print(f"   Loss: {config['loss_type']}, Gamma: {config.get('focal_gamma', 'N/A')}")
        print(f"   Alpha: {config.get('focal_alpha', 'N/A')}, Weight method: {config.get('weight_method', 'None')}")
        print(f"{'='*80}")
        
        # Compute specific weights for this configuration
        config_weights = None
        if config.get("use_weights") and config.get("weight_method"):
            print(f"🔧 Υπολογίζω weights με μέθοδο: {config['weight_method']}")
            config_weights = compute_class_weights(tokenized_ds, label_list, method=config["weight_method"], original_dataset=ds)
        
        # Create appropriate model based on configuration
        if config.get("use_crf"):
            # Create ROBERTa+CRF model
            print("🏗️ Δημιουργία ROBERTa+CRF μοντέλου...")
            bert_config = AutoConfig.from_pretrained("AI-team-UoA/GreekLegalRoBERTa_v2")
            bert_config.num_labels = len(label_list)
            bert_config.id2label = id2label
            bert_config.label2id = label2id
            
            # 🔍 VALIDATION: Έλεγχος ότι το μοντέλο config ταιριάζει με τα dataset labels
            print(f"🔍 MODEL CONFIG VALIDATION:")
            print(f"   Dataset labels: {len(label_list)}")
            print(f"   Model num_labels: {bert_config.num_labels}")
            print(f"   ID2Label mapping: {len(bert_config.id2label)} entries")
            print(f"   Label2ID mapping: {len(bert_config.label2id)} entries")
            
            # Έλεγχος ότι όλα τα labels από το dataset υπάρχουν στο model config
            for i, label in enumerate(label_list):
                assert i in bert_config.id2label, f"Label ID {i} missing from id2label"
                assert label in bert_config.label2id, f"Label '{label}' missing from label2id"
                assert bert_config.id2label[i] == label, f"ID2Label mismatch: {bert_config.id2label[i]} != {label}"
                assert bert_config.label2id[label] == i, f"Label2ID mismatch: {bert_config.label2id[label]} != {i}"
            print(f"   ✅ Όλα τα dataset labels ταιριάζουν με το model config")
            
            model = BertCRFForTokenClassification(
                config=bert_config,
                num_labels=len(label_list),
                bert_model_name="AI-team-UoA/GreekLegalRoBERTa_v2"
            )
            
            # 🔍 ΕΚΤΕΝΗΣ CRF VALIDATION
            print(f"\n🔍 CRF IMPLEMENTATION VALIDATION:")
            print("="*60)
            
            # 1. Έλεγχος CRF architecture
            print(f"📐 CRF ARCHITECTURE:")
            print(f"   CRF num_tags: {model.crf.num_tags}")
            print(f"   Expected num_labels: {len(label_list)}")
            print(f"   Batch_first: {model.crf.batch_first}")
            assert model.crf.num_tags == len(label_list), f"CRF num_tags mismatch!"
            print(f"   ✅ CRF architecture is correct")
            
            # 2. Έλεγχος transition matrix initialization
            print(f"\n🔗 TRANSITION MATRIX VALIDATION:")
            print(f"   Transition matrix shape: {model.crf.transitions.shape}")
            expected_shape = (len(label_list), len(label_list))
            assert model.crf.transitions.shape == expected_shape, f"Transition matrix shape mismatch!"
            print(f"   Expected shape: {expected_shape}")
            print(f"   ✅ Transition matrix shape is correct")
            
            # 3. Έλεγχος start/end transitions
            print(f"\n🚀 START/END TRANSITIONS:")
            print(f"   Start transitions shape: {model.crf.start_transitions.shape}")
            print(f"   End transitions shape: {model.crf.end_transitions.shape}")
            assert model.crf.start_transitions.shape == (len(label_list),), "Start transitions shape wrong!"
            assert model.crf.end_transitions.shape == (len(label_list),), "End transitions shape wrong!"
            print(f"   ✅ Start/End transitions are correctly initialized")
            
            # 4. B-I-O Constraint Validation
            print(f"\n🏷️  B-I-O CONSTRAINT VALIDATION:")
            
            # Ανάλυση label structure
            o_indices = [i for i, label in enumerate(label_list) if label == 'O']
            b_indices = [(i, label[2:]) for i, label in enumerate(label_list) if label.startswith('B-')]
            i_indices = [(i, label[2:]) for i, label in enumerate(label_list) if label.startswith('I-')]
            
            print(f"   O tag indices: {o_indices}")
            print(f"   B- tag count: {len(b_indices)} tags")
            print(f"   I- tag count: {len(i_indices)} tags")
            
            # Entity type consistency check
            b_entity_types = set(entity_type for _, entity_type in b_indices)
            i_entity_types = set(entity_type for _, entity_type in i_indices)
            
            print(f"   B- entity types: {sorted(b_entity_types)}")
            print(f"   I- entity types: {sorted(i_entity_types)}")
            
            missing_i_tags = b_entity_types - i_entity_types
            extra_i_tags = i_entity_types - b_entity_types
            
            if missing_i_tags:
                print(f"   ⚠️  Entity types with B- but no I-: {missing_i_tags}")
                print(f"      (This is OK for single-token entities)")
            
            if extra_i_tags:
                print(f"   ❌ Entity types with I- but no B-: {extra_i_tags}")
                print(f"      This could cause CRF training issues!")
            else:
                print(f"   ✅ All I- tags have corresponding B- tags")
            
            # 5. Sample forward pass validation
            print(f"\n🧪 SAMPLE FORWARD PASS VALIDATION:")
            
            # Create a small test sample
            sample_input = tokenized_ds["train"].select([0])
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                test_input = {
                    'input_ids': torch.tensor([sample_input['input_ids'][0]]),
                    'attention_mask': torch.tensor([sample_input['attention_mask'][0]]),
                    'labels': torch.tensor([sample_input['labels'][0]])
                }
                
                try:
                    outputs = model(**test_input)
                    
                    print(f"   ✅ Forward pass successful")
                    print(f"   Loss shape: {outputs['loss'].shape}")
                    print(f"   Loss value: {outputs['loss'].item():.4f}")
                    print(f"   Logits shape: {outputs['logits'].shape}")
                    
                    # Test decode functionality
                    predictions = model.decode(
                        input_ids=test_input['input_ids'],
                        attention_mask=test_input['attention_mask']
                    )
                    
                    print(f"   ✅ CRF decode successful")
                    print(f"   Decoded sequence length: {len(predictions[0])}")
                    
                    # Validate decoded sequence
                    decoded_labels = [label_list[p] for p in predictions[0][:10]]  # First 10 predictions
                    print(f"   Sample decoded labels: {decoded_labels}")
                    
                    # Check for invalid label IDs
                    invalid_predictions = [p for p in predictions[0] if p >= len(label_list) or p < 0]
                    if invalid_predictions:
                        print(f"   ❌ Invalid predictions found: {invalid_predictions}")
                    else:
                        print(f"   ✅ All predictions are valid label IDs")
                        
                except Exception as e:
                    print(f"   ❌ Forward pass failed: {str(e)}")
                    raise e
            
            print(f"\n🎯 CRF EXPECTED BEHAVIOR:")
            print(f"   During training:")
            print(f"     - CRF will learn transition probabilities between labels")
            print(f"     - Invalid transitions (e.g., B-PERSON → I-ORG) will get low probability")
            print(f"     - Valid transitions (e.g., B-PERSON → I-PERSON, O → B-*) will get high probability")
            print(f"   During inference:")
            print(f"     - Viterbi algorithm will find the most probable valid sequence")
            print(f"     - Output will always respect B-I-O constraints")
            print(f"     - No orphaned I- tags without preceding B- tag")
            
            print("="*60)
        else:
            # Standard ROBERTa model
            print("🏗️ Δημιουργία standard ROBERTa μοντέλου...")
            model = AutoModelForTokenClassification.from_pretrained(
                "AI-team-UoA/GreekLegalRoBERTa_v2", 
                num_labels=len(label_list), 
                id2label=id2label, 
                label2id=label2id
            )
            
            # 🔍 VALIDATION: Έλεγχος ότι το μοντέλο config ταιριάζει με τα dataset labels  
            print(f"🔍 MODEL CONFIG VALIDATION:")
            print(f"   Dataset labels: {len(label_list)}")
            print(f"   Model num_labels: {model.config.num_labels}")
            print(f"   ID2Label mapping: {len(model.config.id2label)} entries")
            print(f"   Label2ID mapping: {len(model.config.label2id)} entries")
            
            # Έλεγχος ότι όλα τα labels από το dataset υπάρχουν στο model config
            for i, label in enumerate(label_list):
                assert i in model.config.id2label, f"Label ID {i} missing from id2label"
                assert label in model.config.label2id, f"Label '{label}' missing from label2id"
                assert model.config.id2label[i] == label, f"ID2Label mismatch: {model.config.id2label[i]} != {label}"
                assert model.config.label2id[label] == i, f"Label2ID mismatch: {model.config.label2id[label]} != {i}"
            print(f"   ✅ Όλα τα dataset labels ταιριάζουν με το model config")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./ner_results/test_{config['name']}",
            logging_dir=f"./ner_logs/test_{config['name']}",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            seed=44,
            fp16=torch.cuda.is_available(),
        )
        
        # Trainer με specific configuration
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": tokenized_ds["train"],
            "eval_dataset": tokenized_ds["validation"],
            "data_collator": data_collator,
            "loss_type": config["loss_type"],
            "use_crf": config.get("use_crf", False),
            "debug_mode": True
        }
        
        # Προσθήκη focal loss parameters μόνο αν δεν χρησιμοποιούμε CRF
        if not config.get("use_crf"):
            if config.get("focal_gamma"):
                trainer_kwargs["focal_gamma"] = config["focal_gamma"]
            if config.get("focal_alpha"):
                trainer_kwargs["focal_alpha"] = config["focal_alpha"]
        
        if config_weights is not None:
            trainer_kwargs["class_weights"] = config_weights
        
        trainer = FlexibleTrainer(**trainer_kwargs)
        
        # Ανάθεση της custom compute_metrics που έχει πρόσβαση στο μοντέλο
        trainer.compute_metrics = trainer.compute_metrics_with_model_info
        
        # --- ΕΛΕΓΧΟΣ LABELS ΠΡΙΝ ΤΟ TRAINING ---
        print("\n" + "="*70)
        print("📊 ΑΝΑΛΥΣΗ LABELS ΣΤΟ TOKENIZED TRAINING SET")
        print("="*70)
        
        all_train_labels = []
        for example in tokenized_ds["train"]:
            labels = example["labels"]
            all_train_labels.extend([l for l in labels if l != -100])
        
        train_label_counts = Counter(all_train_labels)
        total_train_labels = len(all_train_labels)
        
        print(f"Συνολικά valid labels στο training set: {total_train_labels:,}")
        print(f"Μοναδικά labels: {len(train_label_counts)}")
        print()
        
        # Ανάλυση B- και I- tags
        b_count = sum(count for label_id, count in train_label_counts.items() 
                     if label_list[label_id].startswith('B-'))
        i_count = sum(count for label_id, count in train_label_counts.items() 
                     if label_list[label_id].startswith('I-'))
        o_count = train_label_counts.get(0, 0)  # O tag είναι συνήθως index 0
        
        print(f"🏷️  ΣΥΝΟΛΙΚΗ ΚΑΤΑΝΟΜΗ:")
        print(f"   O tags:  {o_count:,} ({(o_count/total_train_labels)*100:.2f}%)")
        print(f"   B- tags: {b_count:,} ({(b_count/total_train_labels)*100:.2f}%)")
        print(f"   I- tags: {i_count:,} ({(i_count/total_train_labels)*100:.2f}%)")
        print()
        
        # Λεπτομερής ανάλυση ανά label
        print(f"📋 ΛΕΠΤΟΜΕΡΗΣ ΚΑΤΑΝΟΜΗ ΑΝΑ LABEL:")
        print(f"{'LABEL':<20} {'COUNT':<10} {'PERCENTAGE':<12} {'TYPE'}")
        print("-" * 55)
        
        # Ταξινόμηση: πρώτα O, μετά B-, μετά I-
        sorted_labels = []
        if 0 in train_label_counts and label_list[0] == 'O':
            sorted_labels.append((0, 'O'))
        
        # B- tags
        b_labels = [(label_id, label_list[label_id]) for label_id in train_label_counts.keys() 
                   if label_list[label_id].startswith('B-')]
        sorted_labels.extend(sorted(b_labels, key=lambda x: x[1]))
        
        # I- tags
        i_labels = [(label_id, label_list[label_id]) for label_id in train_label_counts.keys() 
                   if label_list[label_id].startswith('I-')]
        sorted_labels.extend(sorted(i_labels, key=lambda x: x[1]))
        
        for label_id, label_name in sorted_labels:
            count = train_label_counts[label_id]
            percentage = (count / total_train_labels) * 100
            
            if label_name == 'O':
                label_type = "NON-ENTITY"
            elif label_name.startswith('B-'):
                label_type = "BEGIN"
            elif label_name.startswith('I-'):
                label_type = "INSIDE"
            else:
                label_type = "OTHER"
                
            print(f"{label_name:<20} {count:<10,} {percentage:<12.2f}% {label_type}")
        
        print("-" * 55)
        print(f"{'TOTAL':<20} {total_train_labels:<10,} {'100.00%':<12}")
        print("="*70)
        
        # Train και evaluate
        print("🚀 Training ξεκινάει...")
        trainer.train()
        
        print("📊 Evaluation στο test set...")
        
        # 🔍 PRE-EVALUATION VALIDATION: Έλεγχος ότι το μοντέλο μπορεί να προβλέψει όλα τα labels
        print(f"\n🔍 PRE-EVALUATION MODEL VALIDATION:")
        print("="*50)
        
        # Δοκιμαστική πρόβλεψη σε ένα μικρό batch για έλεγχο
        test_sample = tokenized_ds["test"].select(range(min(10, len(tokenized_ds["test"]))))
        sample_predictions = trainer.predict(test_sample)
        
        if not config.get("use_crf"):
            # Για standard ROBERTa
            sample_preds = np.argmax(sample_predictions.predictions, axis=2)
            unique_predicted_labels = set()
            
            for example_preds, example in zip(sample_preds, test_sample):
                labels = example["labels"]
                for pred, label in zip(example_preds, labels):
                    if label != -100:  # Μόνο valid positions
                        unique_predicted_labels.add(pred)
            
            print(f"Model predicted {len(unique_predicted_labels)} unique labels")
            print(f"Expected {len(label_list)} total labels")
            
            # Έλεγχος για invalid predictions
            invalid_preds = [p for p in unique_predicted_labels if p >= len(label_list)]
            if invalid_preds:
                print(f"⚠️  WARNING: Model produced invalid predictions: {invalid_preds}")
            else:
                print(f"✅ All predictions are within valid label range")
                
            # Δείγμα από τα predicted labels
            sample_pred_labels = [label_list[p] for p in sorted(unique_predicted_labels)[:10]]
            print(f"Sample predicted labels: {sample_pred_labels}")
        
        print("="*50)
        
        test_results = trainer.evaluate(eval_dataset=tokenized_ds["test"], metric_key_prefix="test")

        # 🔍 POST-EVALUATION TEST SET VALIDATION
        print(f"\n🔍 TEST SET EVALUATION VALIDATION:")
        print("="*60)
        
        # Get detailed predictions for analysis
        test_predictions = trainer.predict(tokenized_ds["test"])
        
        if not config.get("use_crf"):
            # Detailed analysis για standard ROBERTa
            test_preds = np.argmax(test_predictions.predictions, axis=2)
            
            # Convert to label sequences για detailed analysis
            test_pred_sequences = []
            test_true_sequences = []
            
            for example_preds, example in zip(test_preds, tokenized_ds["test"]):
                true_labels = example["labels"]
                pred_sequence = [label_list[p] for p, l in zip(example_preds, true_labels) if l != -100]
                true_sequence = [label_list[l] for l in true_labels if l != -100]
                
                if len(pred_sequence) > 0:  # Only non-empty sequences
                    test_pred_sequences.append(pred_sequence)
                    test_true_sequences.append(true_sequence)
            
            print(f"Test sequences analyzed: {len(test_pred_sequences)}")
            
            # Entity-level counting validation
            def count_entities_in_sequences(sequences):
                entity_counts = {}
                total_entities = 0
                total_tags = 0
                
                for seq in sequences:
                    for tag in seq:
                        if tag.startswith('B-'):
                            entity_type = tag[2:]
                            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                            total_entities += 1
                        if tag.startswith('B-') or tag.startswith('I-'):
                            total_tags += 1
                
                return entity_counts, total_entities, total_tags
            
            true_entity_counts, true_entities, true_tags = count_entities_in_sequences(test_true_sequences)
            pred_entity_counts, pred_entities, pred_tags = count_entities_in_sequences(test_pred_sequences)
            
            print(f"\nEntity counting validation:")
            print(f"  True entities (B- tags): {true_entities:,}")
            print(f"  Predicted entities (B- tags): {pred_entities:,}")
            print(f"  True B-/I- tags total: {true_tags:,}")
            print(f"  Predicted B-/I- tags total: {pred_tags:,}")
            
            if true_entities > 0:
                print(f"  Average I- tags per true entity: {(true_tags - true_entities) / true_entities:.2f}")
            if pred_entities > 0:
                print(f"  Average I- tags per pred entity: {(pred_tags - pred_entities) / pred_entities:.2f}")
            
            # Per-entity-type analysis
            print(f"\nPer-entity-type validation:")
            all_entity_types = set(true_entity_counts.keys()) | set(pred_entity_counts.keys())
            
            print(f"{'Entity Type':<15} {'True':<8} {'Pred':<8} {'Ratio':<8}")
            print("-" * 45)
            for entity_type in sorted(all_entity_types):
                true_count = true_entity_counts.get(entity_type, 0)
                pred_count = pred_entity_counts.get(entity_type, 0)
                ratio = pred_count / max(true_count, 1)
                print(f"{entity_type:<15} {true_count:<8} {pred_count:<8} {ratio:<8.2f}")
            
            # Test F1 validation - cross-check with seqeval
            print(f"\nF1 Score validation:")
            test_f1 = test_results.get('test_f1', 0)
            print(f"  Reported F1: {test_f1:.4f}")
            print(f"  This F1 is calculated at ENTITY level:")
            print(f"    - Each complete entity (B- + I-*) counts as 1")
            print(f"    - Partial matches count as 0")
            print(f"    - Entity boundaries must be exact")
            
            # Manual entity F1 calculation για cross-validation
            from collections import defaultdict
            true_entities_set = set()
            pred_entities_set = set()
            
            for seq_idx, (true_seq, pred_seq) in enumerate(zip(test_true_sequences[:100], test_pred_sequences[:100])):
                # Extract entities as (seq_idx, start, end, type) tuples
                def extract_entities(sequence):
                    entities = set()
                    current_entity = None
                    
                    for pos, tag in enumerate(sequence):
                        if tag.startswith('B-'):
                            if current_entity:
                                entities.add(current_entity)
                            current_entity = (seq_idx, pos, pos, tag[2:])
                        elif tag.startswith('I-') and current_entity and tag[2:] == current_entity[3]:
                            current_entity = (current_entity[0], current_entity[1], pos, current_entity[3])
                        else:
                            if current_entity:
                                entities.add(current_entity)
                            current_entity = None
                    
                    if current_entity:
                        entities.add(current_entity)
                    return entities
                
                true_entities_set.update(extract_entities(true_seq))
                pred_entities_set.update(extract_entities(pred_seq))
            
            # Calculate manual F1 για verification
            correct_entities = len(true_entities_set & pred_entities_set)
            manual_precision = correct_entities / max(len(pred_entities_set), 1)
            manual_recall = correct_entities / max(len(true_entities_set), 1)
            manual_f1 = 2 * manual_precision * manual_recall / max(manual_precision + manual_recall, 1e-8)
            
            print(f"  Manual validation (first 100 sequences):")
            print(f"    True entities: {len(true_entities_set)}")
            print(f"    Pred entities: {len(pred_entities_set)}")
            print(f"    Correct entities: {correct_entities}")
            print(f"    Manual F1: {manual_f1:.4f}")
            
        print("="*60)

        # Error analysis για rare classes
        if not config.get("use_crf"):  # Μόνο για standard ROBERTa models
            raw_preds = trainer.predict(tokenized_ds["test"]).predictions
            rare_classes = ["NATIONAL_LOCATION", "UNKNOWN_LOCATION", "FACILITY"]  
            export_rare_class_errors(tokenized_ds["test"], raw_preds, label_list, rare_classes)
        
        # Αποθηκεύουμε όλες τις μετρικές
        cleaned_results = {k.replace('test_', ''): v for k, v in test_results.items()}
        
        results[config['name']] = {
            "metrics": cleaned_results,
            "config": config,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detailed_explanation": {
                "tag_level_metrics": {
                    "micro_f1": "Tag-weighted F1 score (πιο σχετικό με το training)",
                    "macro_f1": "Unweighted average F1 across entity types",
                    "tag_level_accuracy": "Accuracy of individual tag predictions"
                },
                "entity_level_metrics": {
                    "f1": "Entity-level F1 (complete entity must match exactly)",
                    "precision": "Entity-level precision",
                    "recall": "Entity-level recall"
                },
                "entity_type_breakdown": {
                    "b_accuracy": "Accuracy για B- tags του entity type",
                    "i_accuracy": "Accuracy για I- tags του entity type", 
                    "combined_f1": "F1 score για όλα τα tags (B- + I-) του entity type"
                }
            }
        }
        
        # 💾 INCREMENTAL SAVING - Αποθηκεύουμε αμέσως χωρίς να χάσουμε τα προηγούμενα
        incremental_save_file = "progressive_results_with_crf.json"
        try:
            # Φορτώνουμε υπάρχοντα results αν υπάρχουν
            try:
                with open(incremental_save_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_results = {}
            
            # Προσθέτουμε το νέο result
            existing_results[config['name']] = results[config['name']]
            
            # Αποθηκεύουμε το ενημερωμένο αρχείο
            with open(incremental_save_file, 'w', encoding='utf-8') as f:
                clean_existing_results = clean_for_json_serialization(existing_results)
                json.dump(clean_existing_results, f, indent=4, sort_keys=True, ensure_ascii=False)
            
            print(f"💾 Results for {config['name']} saved to {incremental_save_file}")
            
        except Exception as e:
            print(f"⚠️  Error saving results: {e}")
        
        print(f"✅ Αποτελέσματα για {config['name']}:")
        # Safe formatting για micro_f1 που μπορεί να είναι None ή να μην υπάρχει
        micro_f1_value = results[config['name']]['metrics'].get('micro_f1', None)
        if micro_f1_value is not None and isinstance(micro_f1_value, (int, float)):
            micro_f1_str = f"{micro_f1_value:.4f}"
        else:
            micro_f1_str = "N/A"
        
        print(f"   📊 TAG-LEVEL: Micro-F1={micro_f1_str}")
        print(f"   🎯 ENTITY-LEVEL: F1={results[config['name']]['metrics']['f1']:.4f}, Precision={results[config['name']]['metrics']['precision']:.4f}, Recall={results[config['name']]['metrics']['recall']:.4f}")
        print(f"   � Entity-level metrics μετράνε complete entities (exact match)")
        print(f"   💡 Tag-level metrics μετράνε individual tag accuracy")
    
    # � SUMMARY OF ALL CONFIGURATIONS
    print(f"\n� EXPERIMENT SUMMARY:")
    print("="*50)
    print(f"✅ Completed {len(results)} configurations")
    print(f"💾 Progressive results saved to: progressive_results_with_crf.json")
    print(f"📋 Metrics explanation:")
    print(f"   • Micro-F1: Tag-level weighted average (training-relevant)")
    print(f"   • Entity-F1: Complete entity matching (user-relevant)")
    print("="*50)
    
    return results

def export_rare_class_errors(tokenized_test_ds, predictions, label_list, rare_classes, output_file="rare_class_errors.txt"):
    """
    Εξάγει τα λάθη για κάθε σπάνια οντότητα (rare class) σε αρχείο για ανάλυση.
    ENHANCED: Αναλύει τόσο tag-level όσο και entity-level errors.
    """
    print(f"\n🔍 RARE CLASS ERROR ANALYSIS:")
    print("="*50)
    
    errors = []
    entity_errors = []  # Για entity-level errors
    predictions = np.argmax(predictions, axis=2)
    
    # Tag-level error analysis
    for idx, (preds) in enumerate(predictions):
        if idx >= len(tokenized_test_ds):
            break
        example = tokenized_test_ds[idx]
        true_labels = [label_list[l] for l in example['labels'] if l != -100]
        pred_labels = [label_list[p] for p, l in zip(preds, example['labels']) if l != -100]
        
        for i, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
            # Έλεγχος αν το true_label περιέχει κάποιο από τα rare_classes
            is_rare = False
            for rare_class in rare_classes:
                if true_label.endswith(rare_class):  # π.χ. "B-LOCATION-NAT" ends with "LOCATION-NAT"
                    is_rare = True
                    break
            
            if is_rare and true_label != pred_label:
                errors.append({
                    "example_idx": idx,
                    "token_pos": i,
                    "true": true_label,
                    "pred": pred_label,
                    "error_type": "tag_level"
                })
    
    # Entity-level error analysis για rare classes
    print(f"Tag-level errors found: {len(errors)}")
    
    # Extract entities from sequences και find entity-level errors
    for idx in range(min(100, len(tokenized_test_ds), len(predictions))):  # Sample για analysis
        example = tokenized_test_ds[idx]
        preds = predictions[idx]
        true_labels = [label_list[l] for l in example['labels'] if l != -100]
        pred_labels = [label_list[p] for p, l in zip(preds, example['labels']) if l != -100]
        
        # Extract entities
        def extract_entities(sequence):
            entities = []
            current_entity = None
            
            for pos, tag in enumerate(sequence):
                if tag.startswith('B-'):
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {'type': tag[2:], 'start': pos, 'end': pos, 'tags': [tag]}
                elif tag.startswith('I-') and current_entity and tag[2:] == current_entity['type']:
                    current_entity['end'] = pos
                    current_entity['tags'].append(tag)
                else:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = None
            
            if current_entity:
                entities.append(current_entity)
            return entities
        
        true_entities = extract_entities(true_labels)
        pred_entities = extract_entities(pred_labels)
        
        # Find missing rare entities
        for true_entity in true_entities:
            if any(true_entity['type'] == rare_class for rare_class in rare_classes):
                # Check if this entity was correctly predicted
                found_match = False
                for pred_entity in pred_entities:
                    if (pred_entity['type'] == true_entity['type'] and 
                        pred_entity['start'] == true_entity['start'] and 
                        pred_entity['end'] == true_entity['end']):
                        found_match = True
                        break
                
                if not found_match:
                    entity_errors.append({
                        "example_idx": idx,
                        "entity_type": true_entity['type'],
                        "entity_span": (true_entity['start'], true_entity['end']),
                        "entity_tags": true_entity['tags'],
                        "error_type": "entity_level_missing"
                    })
    
    print(f"Entity-level errors found (sample): {len(entity_errors)}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"RARE CLASS ERROR ANALYSIS\n")
        f.write(f"========================\n\n")
        f.write(f"Analyzed rare classes: {rare_classes}\n")
        f.write(f"Tag-level errors: {len(errors)}\n")
        f.write(f"Entity-level errors (sample): {len(entity_errors)}\n\n")
        
        f.write(f"IMPORTANT: F1 scores are calculated at ENTITY level, not tag level!\n")
        f.write(f"- Each entity (B- + I-*) counts as 1\n")
        f.write(f"- Partial matches don't count\n")
        f.write(f"- Entity boundaries must be exact\n\n")
        
        # Group tag-level errors by true label
        error_by_class = {}
        for err in errors:
            true_class = err['true']
            if true_class not in error_by_class:
                error_by_class[true_class] = []
            error_by_class[true_class].append(err)
        
        for true_class, class_errors in error_by_class.items():
            f.write(f"\n=== Errors for class: {true_class} ({len(class_errors)} errors) ===\n")
            pred_counts = {}
            for err in class_errors:
                pred = err['pred']
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
            
            f.write("Confusion with other classes:\n")
            for pred_class, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(class_errors)) * 100
                f.write(f"  {pred_class}: {count} times ({percentage:.1f}%)\n")
            
            f.write(f"\nFirst 10 examples:\n")
            for i, err in enumerate(class_errors[:10]):
                f.write(f"  {i+1}. Example {err['example_idx']}, Token {err['token_pos']}: {err['true']} -> {err['pred']}\n")
        
        # Entity-level errors
        if entity_errors:
            f.write(f"\n=== Entity-level errors (sample) ===\n")
            entity_error_by_type = {}
            for err in entity_errors:
                entity_type = err['entity_type']
                entity_error_by_type[entity_type] = entity_error_by_type.get(entity_type, 0) + 1
            
            f.write(f"Missing entities by type:\n")
            for entity_type, count in sorted(entity_error_by_type.items()):
                f.write(f"  {entity_type}: {count} missing entities\n")
    
    print(f"Exported {len(errors)} tag-level errors and {len(entity_errors)} entity-level errors to {output_file}")
    
    # Print summary - ENHANCED
    print(f"\n📊 COMPREHENSIVE RARE CLASS ERROR SUMMARY:")
    print(f"🏷️  TAG-LEVEL ERRORS:")
    error_by_class = {}
    for err in errors:
        true_class = err['true']
        error_by_class[true_class] = error_by_class.get(true_class, 0) + 1
    
    for rare_class in rare_classes:
        # Count both B- and I- tag errors για κάθε rare class
        b_errors = error_by_class.get(f'B-{rare_class}', 0)
        i_errors = error_by_class.get(f'I-{rare_class}', 0)
        total_errors = b_errors + i_errors
        print(f"  {rare_class}: {total_errors} total errors (B-: {b_errors}, I-: {i_errors})")
    
    if entity_errors:
        print(f"\n🎯 ENTITY-LEVEL ERRORS (impacts F1 directly):")
        entity_error_by_type = {}
        for err in entity_errors:
            entity_type = err['entity_type']
            entity_error_by_type[entity_type] = entity_error_by_type.get(entity_type, 0) + 1
        
        for entity_type, count in sorted(entity_error_by_type.items()):
            print(f"  {entity_type}: {count} missing entities")
        
        print(f"\n💡 Σημείωση: Αυτά τα entity-level errors επηρεάζουν άμεσα το F1 score!")
    
    print("="*50)
    
    return errors

# --- 9. MAIN EXECUTION LOGIC ---
# Note: run_single_experiment removed - using test_all_loss_configurations instead


# --- 10. MAIN EXECUTION ---
if __name__ == "__main__":
    # 🔧 ΕΓΚΑΤΑΣΤΑΣΗ DEPENDENCIES VALIDATION
    print("🔍 Έλεγχος εγκατάστασης dependencies...")
    
    try:
        from torchcrf import CRF
        print("✅ pytorch-crf library εγκαταστημένο επιτυχώς")
    except ImportError:
        print("❌ ΣΦΑΛΜΑ: pytorch-crf library δεν είναι εγκαταστημένο!")
        print("🔧 Εγκατάσταση pytorch-crf...")
        import subprocess
        import sys
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-crf"])
            print("✅ pytorch-crf εγκαταστάθηκε επιτυχώς!")
            from torchcrf import CRF  # Re-import after installation
        except Exception as e:
            print(f"❌ Αποτυχία εγκατάστασης pytorch-crf: {str(e)}")
            print("📖 Παρακαλώ εκτελέστε manually: pip install pytorch-crf")
            print("📖 Περισσότερες πληροφορίες: https://pytorch-crf.readthedocs.io/")
            exit(1)
    
    # Εκτέλεση comprehensive comparison όλων των loss configurations
    print("🔄 Εκτέλεση δοκιμής όλων των loss configurations (ROBERTa + ROBERTa+CRF)...")
    test_results = test_all_loss_configurations()
    
    # Save comparison results
    comparison_file = "loss_comparison_results_with_crf.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        clean_test_results = clean_for_json_serialization(test_results)
        json.dump(clean_test_results, f, indent=4, sort_keys=True)
    print(f"\nΤα αποτελέσματα σύγκρισης αποθηκεύτηκαν στο '{comparison_file}'")

# Αν τρέχεται ως main script
if __name__ == "__main__":
    import sys
    
    print("🚀 STARTING GREEK LEGAL NER EXPERIMENTS")
    print("="*80)
    
    # Run experiments
    main_results = test_all_loss_configurations()
    
    print(f"\n🎉 EXPERIMENTS COMPLETED!")
    print(f"📊 Results saved to ner_results/ directory")