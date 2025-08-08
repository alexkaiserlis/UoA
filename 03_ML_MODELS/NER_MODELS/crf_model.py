"""
CRF Model Implementation for Greek Legal NER

This module contains the BertCRFForTokenClassification class which combines
ROBERTa with a CRF layer for improved Named Entity Recognition performance.

🔍 CRF VALIDATION & EXPLANATION:
================================
Το CRF (Conditional Random Field) layer προσθέτει structural constraints:

🎯 ΣΕ ΤΙ ΒΑΣΙΖΕΤΑΙ ΤΟ CRF ΓΙΑ ΣΥΣΧΕΤΙΣΜΟΥΣ:
============================================

1. **BIO TAGGING CONSTRAINTS** (Κύρια Βάση):
   - Βασίζεται στη δομή του BIO scheme (Begin-Inside-Outside)
   - Μαθαίνει τις ΕΠΙΤΡΕΠΤΕΣ μεταβάσεις μεταξύ tags:
     ✅ B-PERSON → I-PERSON (συνέχεια entity)
     ✅ B-PERSON → O (τέλος entity)
     ✅ O → B-PERSON (αρχή νέου entity)
     ❌ B-PERSON → I-ORG (αλλαγή τύπου χωρίς τέλος)
     ❌ O → I-PERSON (I- χωρίς προηγούμενο B-)
     ❌ I-PERSON → I-ORG (άμεση αλλαγή τύπου)

2. **SEQUENTIAL DEPENDENCIES** (Αλληλουχία):
   - Το CRF μοντελοποιεί P(y_i | y_{i-1}, x_sequence)
   - Όχι απλώς P(y_i | x_i) όπως κάνει το ROBERTa μόνο του
   - Λαμβάνει υπόψη ΟΛΗ την αλληλουχία για κάθε απόφαση

3. **EMISSION PROBABILITIES** (από ROBERTa):
   - ROBERTa logits = πόσο πιθανό είναι κάθε label για κάθε token
   - Βασίζεται σε: context, word embeddings, attention mechanisms
   - ΔΕΝ χρησιμοποιεί POS tags ή άλλα linguistic features εκτός contextual embeddings

4. **TRANSITION PROBABILITIES** (Μαθημένες):
   - Το CRF μαθαίνει πίνακα μεταβάσεων [num_labels × num_labels]
   - transitions[i][j] = score για μετάβαση από label i σε label j
   - Αυτές οι πιθανότητες βασίζονται ΜΟΝΟ στο BIO scheme και τα training data

🔬 ΤΕΧΝΙΚΕΣ ΛΕΠΤΟΜΕΡΕΙΕΣ ΣΥΣΧΕΤΙΣΜΩΝ:
=====================================

A. **FEATURE FUNCTIONS** που χρησιμοποιεί:
   - f1(y_i, x, i): Emission features (ROBERTa representations)
   - f2(y_{i-1}, y_i): Transition features (label bigrams)
   - f3(y_0): Start state features
   - f4(y_n): End state features

B. **ΔΕΝ χρησιμοποιεί**:
   ❌ POS tags (Part-of-Speech tagging)
   ❌ Syntactic parse trees
   ❌ External linguistic features
   ❌ Gazetteer lookups
   ❌ Character-level features (beyond what ROBERTa already captures)

C. **Βασίζεται ΜΟΝΟ σε**:
   ✅ BIO tagging structure
   ✅ ROBERTa contextual embeddings
   ✅ Sequential label dependencies
   ✅ Training data patterns

🧠 ΜΑΘΗΣΙΑΚΗ ΔΙΑΔΙΚΑΣΙΑ:
=======================
1. ROBERTa μαθαίνει representations που encoding context, semantics, syntax
2. CRF μαθαίνει ποιες μεταβάσεις είναι valid στο BIO scheme
3. Viterbi algorithm βρίσκει την καλύτερα structured sequence

📊 ΠΑΡΑΔΕΙΓΜΑ ΣΥΣΧΕΤΙΣΜΟΥ:
=========================
Input: "Ο Γιάννης Παπαδόπουλος εργάζεται στον ΟΤΕ"

ROBERTa alone: ["O", "B-PERSON", "I-ORG", "O", "O", "B-ORG"]  ❌ (invalid!)
ROBERTa + CRF: ["O", "B-PERSON", "I-PERSON", "O", "O", "B-ORG"] ✅ (valid!)

Το CRF διόρθωσε το "I-ORG" → "I-PERSON" γιατί:
- Δεν επιτρέπεται I-ORG μετά από B-PERSON
- Το transition model έμαθε ότι B-PERSON → I-PERSON είναι πιο πιθανό

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

import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF


class BertCRFForTokenClassification(nn.Module):
    """
    ROBERTa + CRF μοντέλο για Token Classification
    
    Combines the contextual representations from ROBERTa with the sequence
    modeling capabilities of CRF to ensure valid BIO tag transitions.
    
    Args:
        config: Model configuration object
        num_labels (int): Number of labels in the label vocabulary
        bert_model_name (str): Name of the pre-trained BERT/ROBERTa model
    
    Returns:
        Dict containing:
        - logits: Raw token classification logits
        - loss: CRF loss (if labels provided)
        
    Example:
        >>> config = AutoConfig.from_pretrained("AI-team-UoA/GreekLegalRoBERTa_v2")
        >>> model = BertCRFForTokenClassification(config, num_labels=19)
        >>> outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        >>> loss = outputs["loss"]
        >>> logits = outputs["logits"]
    """
    
    def __init__(self, config, num_labels, bert_model_name="AI-team-UoA/GreekLegalRoBERTa_v2"):
        super(BertCRFForTokenClassification, self).__init__()
        self.num_labels = num_labels
        self.config = config
        
        # Load ROBERTa model
        self.bert = AutoModel.from_pretrained(bert_model_name, config=config)
        
        # Dropout layer
        dropout_prob = getattr(config, 'hidden_dropout_prob', 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Classification layer: ROBERTa hidden states → label logits
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # CRF layer for sequence labeling
        self.crf = CRF(num_labels, batch_first=True)
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize the classifier layer weights using normal distribution.
        
        This follows the standard practice for transformer fine-tuning where
        the classification head is initialized with small random weights.
        """
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass of the ROBERTa + CRF model.
        
        Args:
            input_ids (torch.Tensor): Token IDs of shape [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len]
            token_type_ids (torch.Tensor, optional): Token type IDs (not used for ROBERTa)
            labels (torch.Tensor, optional): Ground truth labels for training
        
        Returns:
            Dict containing:
            - logits: Token classification logits [batch_size, seq_len, num_labels]
            - loss: CRF loss (only if labels are provided)
        
        Note:
            During training, labels should be provided to compute CRF loss.
            Labels with value -100 are ignored (padding tokens).
        """
        # ROBERTa encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Extract sequence representations
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)
        
        # Classification: hidden states → label logits
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]
        
        # Prepare outputs
        model_outputs = {"logits": logits}
        
        # Compute CRF loss if labels are provided
        if labels is not None:
            loss = self._compute_crf_loss(logits, labels, attention_mask)
            model_outputs["loss"] = loss
        
        return model_outputs
    
    def _compute_crf_loss(self, logits, labels, attention_mask):
        """
        Compute CRF loss given logits and labels.
        
        Args:
            logits (torch.Tensor): Token classification logits
            labels (torch.Tensor): Ground truth labels
            attention_mask (torch.Tensor): Attention mask
        
        Returns:
            torch.Tensor: CRF loss value
        
        Note:
            - Converts -100 labels to 0 (O tag) since CRF cannot handle -100
            - Creates proper mask to ignore padding tokens
        """
        # Create attention mask for CRF (exclude padding tokens)
        if attention_mask is not None:
            crf_mask = attention_mask.bool()
        else:
            crf_mask = torch.ones_like(input_ids).bool()
        
        # Convert -100 labels to 0 (O tag) for CRF compatibility
        # CRF cannot handle -100 values, so we map them to the O tag
        crf_labels = labels.clone()
        crf_labels[crf_labels == -100] = 0  # Map to O tag (index 0)
        
        # Update mask to ignore padding tokens in loss computation
        crf_mask = crf_mask & (labels != -100)
        
        # CRF loss: negative log-likelihood of the true sequence
        loss = -self.crf(logits, crf_labels, mask=crf_mask, reduction='mean')
        
        return loss
    
    def decode(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Decode the best sequence using CRF Viterbi algorithm.
        
        This method should be used during inference to get the most likely
        sequence of labels according to the CRF model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            token_type_ids (torch.Tensor, optional): Token type IDs
        
        Returns:
            List[List[int]]: Predicted label sequences for each sample in batch
        
        Example:
            >>> predictions = model.decode(input_ids, attention_mask)
            >>> # predictions[0] contains the predicted labels for the first sample
        
        Note:
            This method uses Viterbi decoding to find the most probable
            sequence that respects BIO tagging constraints.
        """
        # Get logits from forward pass (without labels)
        outputs = self.forward(input_ids, attention_mask, token_type_ids)
        logits = outputs["logits"]
        
        # Create mask for CRF decoding
        if attention_mask is not None:
            crf_mask = attention_mask.bool()
        else:
            crf_mask = torch.ones_like(input_ids).bool()
        
        # CRF Viterbi decoding to get the best sequence
        predictions = self.crf.decode(logits, mask=crf_mask)
        
        return predictions
    
    def get_crf_transitions(self):
        """
        Get the learned CRF transition matrix.
        
        Returns:
            torch.Tensor: Transition matrix of shape [num_labels, num_labels]
                         where transitions[i][j] represents the score for
                         transitioning from label i to label j.
        
        Note:
            This can be useful for analyzing what transitions the model
            has learned during training.
        """
        return self.crf.transitions
    
    def freeze_bert_layers(self, num_layers_to_freeze=None):
        """
        Freeze BERT layers to prevent updates during training.
        
        Args:
            num_layers_to_freeze (int, optional): Number of layers to freeze from bottom.
                                                 If None, freezes all BERT parameters.
        
        Example:
            >>> model.freeze_bert_layers(6)  # Freeze first 6 layers
            >>> model.freeze_bert_layers()   # Freeze all BERT layers
        """
        if num_layers_to_freeze is None:
            # Freeze all BERT parameters
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            # Freeze specific number of layers
            layers_to_freeze = self.bert.encoder.layer[:num_layers_to_freeze]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def unfreeze_bert_layers(self):
        """
        Unfreeze all BERT layers to allow gradient updates.
        """
        for param in self.bert.parameters():
            param.requires_grad = True


def create_crf_model(num_labels, model_name="AI-team-UoA/GreekLegalRoBERTa_v2", config=None):
    """
    Factory function to create a CRF model with proper configuration.
    
    Args:
        num_labels (int): Number of labels in the classification task
        model_name (str): Name of the pre-trained model to use
        config: Optional model configuration
    
    Returns:
        BertCRFForTokenClassification: Initialized model
    
    Example:
        >>> model = create_crf_model(num_labels=19)
        >>> # Model is ready for training or inference
    """
    if config is None:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
    
    model = BertCRFForTokenClassification(config, num_labels, model_name)
    return model


# Model configuration constants
DEFAULT_MODEL_NAME = "AI-team-UoA/GreekLegalRoBERTa_v2"
DEFAULT_DROPOUT_PROB = 0.1
DEFAULT_WEIGHT_INIT_STD = 0.02

__all__ = [
    'BertCRFForTokenClassification',
    'create_crf_model',
    'DEFAULT_MODEL_NAME',
    'DEFAULT_DROPOUT_PROB',
    'DEFAULT_WEIGHT_INIT_STD'
]
