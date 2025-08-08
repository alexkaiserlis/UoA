# 🔗 CRF Ρυθμίσεις & Υλοποίηση στο Greek Legal NER

## 📋 Τι είναι το CRF (Conditional Random Field)

Το **CRF** είναι ένα **probabilistic model** που εξασφαλίζει **valid sequences** στο Named Entity Recognition. Ενώ το ROBERTa προβλέπει κάθε token ανεξάρτητα, το CRF λαμβάνει υπόψη **όλη την αλληλουχία** για να δώσει structurally consistent predictions.

## ⚙️ Οι Παράμετροι που Έχετε Βάλει

### 🎯 **Κύριες CRF Ρυθμίσεις**

```python
# Στο crf_model.py:
self.crf = CRF(
    num_labels,           # 17 labels (8 entities × 2 + O)
    batch_first=True      # 🎯 Tensor format: [batch_size, seq_len]
)

# Στο config.py:
crf_reduction: str = "mean"    # 🎯 Loss aggregation method
use_crf: bool = True           # 🎯 Enable CRF layer
```

### 🔧 **CRF Loss Computation**

```python
# Στο crf_model.py γραμμή ~237:
loss = -self.crf(
    logits,                    # ROBERTa predictions [batch, seq_len, num_labels]
    crf_labels,               # True labels [batch, seq_len]  
    mask=crf_mask,            # Mask για padding tokens
    reduction='mean'          # 🎯 Average loss across batch
)
```

### 🎭 **Mask Handling**

```python
# Proper padding handling:
crf_labels = labels.clone()
crf_labels[crf_labels == -100] = 0  # Map padding to O tag (index 0)
crf_mask = attention_mask.bool() & (labels != -100)  # Ignore padding
```

---

## 🏗️ Πώς Δουλεύει το CRF (Βήμα-βήμα)

### Βήμα 1: ROBERTa Emissions
```python
# ROBERTa δίνει "emission scores" για κάθε token:
text = ["Ο", "Γιάννης", "εργάζεται", "στη", "Microsoft"]

emission_scores = [
    [0.9, 0.05, 0.05, ...],    # "Ο": υψηλό score για "O"
    [0.1, 0.8, 0.05, 0.05],    # "Γιάννης": υψηλό για "B-PERSON"  
    [0.85, 0.1, 0.05, ...],    # "εργάζεται": υψηλό για "O"
    [0.9, 0.05, 0.05, ...],    # "στη": υψηλό για "O"
    [0.1, 0.05, 0.05, 0.8]     # "Microsoft": υψηλό για "B-ORG"
]
```

### Βήμα 2: CRF Transition Matrix
```python
# Το CRF μαθαίνει πίνακα μεταβάσεων [17×17]:
transition_scores = {
    # Valid transitions (υψηλά scores):
    "O → B-PERSON": 2.3,         # ✅ Μπορεί να ξεκινήσει entity
    "B-PERSON → I-PERSON": 3.1,  # ✅ Συνέχεια entity
    "B-PERSON → O": 1.8,         # ✅ Τέλος entity
    
    # Invalid transitions (χαμηλά scores):
    "O → I-PERSON": -5.2,        # ❌ I- χωρίς B-
    "B-PERSON → I-ORG": -4.8,    # ❌ Αλλαγή entity type
    "I-PERSON → B-PERSON": -2.1, # ❌ Νέο entity χωρίς O
}
```

### Βήμα 3: Sequence Scoring
```python
# Για κάθε πιθανή sequence, το CRF υπολογίζει:
# Score(sequence) = Σ(emission_scores) + Σ(transition_scores)

# Candidate 1: ["O", "B-PERSON", "O", "O", "B-ORG"]
score_1 = emission["O"][0] + transition["START→O"] +
          emission["B-PERSON"][1] + transition["O→B-PERSON"] +
          emission["O"][2] + transition["B-PERSON→O"] +
          emission["O"][3] + transition["O→O"] +
          emission["B-ORG"][4] + transition["O→B-ORG"] +
          transition["B-ORG→END"]

# Candidate 2: ["O", "B-PERSON", "I-PERSON", "O", "B-ORG"] ❌ (wrong boundaries)
score_2 = emission["O"][0] + transition["START→O"] +
          emission["B-PERSON"][1] + transition["O→B-PERSON"] +
          emission["I-PERSON"][2] + transition["B-PERSON→I-PERSON"] +  # Lower emission!
          emission["O"][3] + transition["I-PERSON→O"] +
          emission["B-ORG"][4] + transition["O→B-ORG"]

# Το CRF επιλέγει την sequence με το υψηλότερο συνολικό score
```

### Βήμα 4: Viterbi Decoding
```python
# Viterbi algorithm βρίσκει την καλύτερη sequence efficiently:
best_sequence = crf.decode(logits, mask=crf_mask)
# Αποτέλεσμα: ["O", "B-PERSON", "O", "O", "B-ORG"]
```

---

## 🎯 Αναλυτικές Παράμετροι στον Κώδικά σας

### 1. **`num_labels=17`**
```python
# Τα 17 labels σας:
labels = [
    "O",                      # 0: Outside
    "B-FACILITY",            # 1: Begin Facility
    "I-FACILITY",            # 2: Inside Facility  
    "B-GPE",                 # 3: Begin Geo-Political Entity
    "I-GPE",                 # 4: Inside GPE
    "B-LEGISLATION_REFERENCE", # 5: Begin Legislation Reference
    "I-LEGISLATION_REFERENCE", # 6: Inside Legislation Reference
    "B-NATIONAL_LOCATION",     # 7: Begin National Location
    "I-NATIONAL_LOCATION",     # 8: Inside National Location
    "B-UNKNOWN_LOCATION",      # 9: Begin Unknown Location
    "I-UNKNOWN_LOCATION",      # 10: Inside Unknown Location
    "B-ORGANIZATION",          # 11: Begin Organization
    "I-ORGANIZATION",          # 12: Inside Organization
    "B-PERSON",               # 13: Begin Person
    "I-PERSON",               # 14: Inside Person
    "B-PUBLIC_DOCUMENT",       # 15: Begin Public Document
    "I-PUBLIC_DOCUMENT"        # 16: Inside Public Document
]

# CRF transition matrix: 17×17 = 289 παράμετροι να μάθει
```

### 2. **`batch_first=True`**
```python
# Tensor dimensions:
logits.shape = [batch_size, sequence_length, num_labels]
labels.shape = [batch_size, sequence_length]

# Αντί για [sequence_length, batch_size, num_labels] (default)
# Συμβατό με τα PyTorch/HuggingFace conventions
```

### 3. **`reduction='mean'`**
```python
# Loss aggregation options:

# 'mean' (Current choice):
final_loss = sum(sequence_losses) / num_valid_sequences
# Κανονικοποιημένο - καλό για comparison across batches

# 'sum' (Alternative):  
final_loss = sum(sequence_losses)
# Μη-κανονικοποιημένο - υψηλότερες τιμές

# 'none' (Alternative):
final_loss = [loss_seq1, loss_seq2, ...]  # Tensor με individual losses
# Για custom weighting
```

### 4. **Mask Handling**
```python
# Η στρατηγική σας για padding:

# Step 1: Map -100 to O tag (index 0)
crf_labels[crf_labels == -100] = 0

# Step 2: Create proper mask  
crf_mask = attention_mask.bool() & (labels != -100)

# Step 3: Apply mask in CRF
loss = -self.crf(logits, crf_labels, mask=crf_mask, reduction='mean')

# ✅ Εξαιρεί padding tokens από loss computation
# ✅ Δεν επηρεάζει transition learning από invalid positions
```

---

## 🔄 Εναλλακτικές Ρυθμίσεις & Options

### 1. **Reduction Strategies**

#### Option A: `reduction='sum'` (Cumulative Loss)
```python
# Στο config.py:
crf_reduction: str = "sum"

# Στο crf_model.py: 
loss = -self.crf(logits, crf_labels, mask=crf_mask, reduction='sum')

# ✅ Καλό για: Variable batch sizes, gradient accumulation
# ❌ Πρόβλημα: Non-normalized loss values
```

#### Option B: `reduction='none'` (Per-Sequence Loss)
```python
# Στο config.py:
crf_reduction: str = "none"

# Στο crf_model.py:
sequence_losses = -self.crf(logits, crf_labels, mask=crf_mask, reduction='none')
# sequence_losses.shape = [batch_size]

# Custom weighting:
weights = torch.tensor([1.0, 2.0, 1.5, ...])  # Per-sequence weights
weighted_loss = (sequence_losses * weights).mean()

# ✅ Καλό για: Custom weighting, sample importance
# ❌ Πρόβλημα: Πιο πολύπλοκη implementation
```

### 2. **CRF Constraint Variants**

#### Option A: **Linear-Chain CRF** (Current)
```python
# Τι έχετε τώρα:
self.crf = CRF(num_labels, batch_first=True)

# Μοντελοποιεί: P(y_i | y_{i-1}, x)
# Μόνο bigram transitions: label[i-1] → label[i]
```

#### Option B: **Semi-CRF** (Advanced)
```python
# Εναλλακτικό (δεν το έχετε):
from torchcrf import SemiCRF
self.crf = SemiCRF(num_labels, max_seg_len=10)

# Μοντελοποιεί: P(segment | x)  
# Μαθαίνει entity lengths, όχι μόνο transitions
# ✅ Καλύτερο για entities με specific length patterns
# ❌ Πιο πολύπλοκο, αργότερο training
```

#### Option C: **Constrained CRF** (Custom)
```python
# Custom constraints (δεν το έχετε):
def create_constraint_matrix(num_labels):
    """Manually define valid transitions"""
    transitions = torch.full((num_labels, num_labels), -1000.0)  # Very low score
    
    # Allow only valid BIO transitions:
    transitions[0, 0] = 0.0    # O → O
    transitions[0, 1::2] = 0.0  # O → B-* (all B- tags)
    
    for i in range(1, num_labels, 2):  # B- tags  
        transitions[i, 0] = 0.0      # B-* → O
        transitions[i, i+1] = 0.0    # B-* → I-* (same entity)
        transitions[i, 1::2] = 0.0   # B-* → B-* (new entity)
    
    for i in range(2, num_labels, 2):  # I- tags
        transitions[i, 0] = 0.0      # I-* → O  
        transitions[i, i] = 0.0      # I-* → I-* (continue)
        transitions[i, 1::2] = 0.0   # I-* → B-* (new entity)
    
    return transitions

# ✅ Εξαιρετικά αυστηρό BIO enforcement
# ❌ Λιγότερη flexibility για το μοντέλο να μάθει
```

### 3. **Alternative Sequence Models**

#### Option A: **BiLSTM-CRF** (Classic)
```python
# Αντί για ROBERTa + CRF:
class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                           bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

# ✅ Ταχύτερο training, λιγότερα parameters
# ❌ Χειρότερες representations από ROBERTa
```

#### Option B: **Transformer without CRF** (Simple)
```python
# Μόνο ROBERTa χωρίς CRF:
class SimpleTransformer(nn.Module):
    def __init__(self, model_name, num_labels):
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

# ✅ Απλούστερο, ταχύτερο inference  
# ❌ Δεν εξασφαλίζει valid BIO sequences
```

#### Option C: **Pointer Networks** (Advanced)
```python
# Sequence-to-sequence approach:
# Αντί για token classification → span detection
# Μαθαίνει start/end positions για κάθε entity

# ✅ Καλύτερο για overlapping entities
# ❌ Πιο πολύπλοκο, χρειάζεται διαφορετικό data format
```

### 4. **Loss Function Variants**

#### Option A: **Focal CRF Loss** (Custom)
```python
# Συνδυασμός CRF + Focal Loss:
class FocalCRFLoss(nn.Module):
    def __init__(self, crf, gamma=2.0, alpha=None):
        self.crf = crf
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, labels, mask):
        # Standard CRF loss
        crf_loss = -self.crf(logits, labels, mask=mask, reduction='none')
        
        # Apply focal weighting
        pt = torch.exp(-crf_loss)  # "confidence"
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_weight = self.alpha[labels]  # Per-class weighting
            focal_weight = alpha_weight * focal_weight
        
        return (focal_weight * crf_loss).mean()

# ✅ Combines structure (CRF) + class balance (Focal)
# ❌ Πιο πολύπλοκο, πειραματικό
```

#### Option B: **Weighted CRF Loss** (Custom)
```python
# Per-sequence weighting:
def weighted_crf_loss(crf, logits, labels, mask, sequence_weights):
    sequence_losses = -crf(logits, labels, mask=mask, reduction='none')
    weighted_losses = sequence_losses * sequence_weights
    return weighted_losses.mean()

# ✅ Καλό για domain adaptation, temporal weighting
# ❌ Χρειάζεται manual weight definition
```

---

## 🎯 Συνιστώμενες Ρυθμίσεις για σας

### ✅ **Κρατήστε τις Υπάρχουσες** (Είναι άριστες!)

```python
# Τρέχουσες ρυθμίσεις (PERFECT για research):
self.crf = CRF(num_labels=17, batch_first=True)
loss = -self.crf(logits, crf_labels, mask=crf_mask, reduction='mean')
crf_reduction: str = "mean"
```

**Γιατί**:
- **Standard approach**: Αυτό που χρησιμοποιούν τα top NER papers
- **Proper masking**: Σωστό handling των padding tokens
- **Efficient**: Optimal balance performance vs complexity

### 🔄 **Εναλλακτικές για Πειραματισμό**

#### Experiment 1: Different Reduction
```python
# Στο config.py, προσθέστε:
crf_reduction: str = "sum"  # Αντί για "mean"

# Test impact on:
# - Training stability
# - Convergence speed  
# - Final performance
```

#### Experiment 2: Constrained Transitions
```python
# Προσθέστε στο crf_model.py:
def initialize_transitions(self):
    """Initialize CRF transitions with BIO constraints"""
    with torch.no_grad():
        # Set invalid transitions to very low values
        self.crf.transitions.fill_(-1000.0)
        
        # Allow valid transitions
        for i in range(self.num_labels):
            for j in range(self.num_labels):
                if self.is_valid_transition(i, j):
                    self.crf.transitions[i, j] = 0.0

# ✅ Enforces strict BIO compliance από την αρχή
# Test: Does it help with rare entities?
```

#### Experiment 3: Per-Entity CRF Weighting
```python
# Custom loss για rare entities:
def entity_weighted_crf_loss(crf, logits, labels, mask, entity_weights):
    # Weight sequences based on rare entities they contain
    sequence_weights = compute_sequence_weights(labels, entity_weights)
    sequence_losses = -crf(logits, labels, mask=mask, reduction='none')
    return (sequence_losses * sequence_weights).mean()

# ✅ Focus περισσότερο στα rare entity sequences
```

---

## 🔍 Monitoring & Debugging CRF

### 📊 **CRF-Specific Metrics να Παρακολουθείτε**

```python
# Προσθέστε στο evaluation:

def analyze_crf_performance(self, model, dataset):
    """Analyze CRF-specific performance"""
    
    # 1. BIO Constraint Violations
    violations = count_bio_violations(predictions)
    print(f"BIO violations: {violations} / {total_predictions}")
    
    # 2. Transition Matrix Analysis  
    transitions = model.crf.transitions.detach()
    valid_transitions = extract_high_scoring_transitions(transitions)
    
    # 3. Entity Boundary Accuracy
    boundary_accuracy = compute_boundary_accuracy(true_entities, pred_entities)
    
    # 4. Sequence-Level Accuracy (complete match)
    sequence_accuracy = compute_sequence_accuracy(true_sequences, pred_sequences)
    
    return {
        'bio_violation_rate': violations / total_predictions,
        'boundary_accuracy': boundary_accuracy,
        'sequence_accuracy': sequence_accuracy,
        'learned_transitions': valid_transitions
    }
```

### 🎯 **Success Indicators**

```python
# Καλά signs για CRF:
✅ BIO violation rate < 1%
✅ Boundary accuracy > 85%  
✅ Sequence accuracy improvements vs non-CRF
✅ Learned transitions make linguistic sense

# Red flags:
❌ High BIO violation rate (CRF not learning constraints)
❌ No improvement over standard CrossEntropy
❌ Erratic transition matrix (numerical instability)
```

**Bottom line**: Οι CRF ρυθμίσεις σας είναι **research-grade standard**. Το μόνο που θα δοκιμάζατε είναι διαφορετικό `reduction` method, αλλά το `"mean"` είναι η καλύτερη επιλογή για σταθερό training! 🎯
