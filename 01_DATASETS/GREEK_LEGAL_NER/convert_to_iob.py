#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script για μετατροπή των αριθμητικών JSON datasets σε IOB format
Συμβαδίζει με τη λογική του count_labels.py
"""

import json
import os

def load_label_mapping(mapping_file):
    """
    Φορτώνει το mapping αριθμών σε ονόματα labels
    """
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        # Αντιστροφή του mapping (από όνομα -> αριθμός σε αριθμός -> όνομα)
        reverse_mapping = {str(v): k for k, v in mapping.items()}
        return reverse_mapping
    except Exception as e:
        print(f"Σφάλμα φόρτωσης mapping: {e}")
        return {}

def convert_numeric_to_iob_labels(labels_list, label_mapping):
    """
    Μετατρέπει αριθμητικά labels σε IOB format με ονόματα
    ΙΔΙΑ ΛΟΓΙΚΗ ΜΕ count_labels.py
    
    Args:
        labels_list (list): Lista με τα αριθμητικά labels μιας πρότασης
        label_mapping (dict): Mapping από αριθμό σε όνομα
        
    Returns:
        list: Lista με IOB tags με ονόματα
    """
    iob_tags = []
    
    for i, label in enumerate(labels_list):
        label_str = str(label)
        
        if label_str == '0':
            iob_tags.append('O')
        else:
            # Παίρνουμε το όνομα του label από το mapping
            label_name = label_mapping.get(label_str, f'UNKNOWN_{label}')
            
            # Αν το label ήδη περιέχει B- ή I-, το κρατάμε όπως είναι
            if label_name.startswith('B-') or label_name.startswith('I-'):
                iob_tags.append(label_name)
            else:
                # Αλλιώς, καθορίζουμε αν είναι B- ή I- βάσει της θέσης
                if i == 0 or str(labels_list[i-1]) != label_str:
                    iob_tags.append(f'B-{label_name}')
                else:
                    iob_tags.append(f'I-{label_name}')
    
    return iob_tags

def convert_dataset_to_iob(input_file, output_file, label_mapping):
    """
    Μετατρέπει ένα dataset από αριθμητικά labels σε IOB format
    """
    print(f"Μετατρέπω {input_file} -> {output_file}")
    
    # Φόρτωση του αρχικού dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Εξαγωγή δεδομένων
    inputs = data.get('input', [])
    numeric_labels = data.get('label', [])
    languages = data.get('language', ['el'] * len(inputs))  # Default σε ελληνικά αν δεν υπάρχει
    
    print(f"  - Βρέθηκαν {len(inputs)} προτάσεις")
    print(f"  - Βρέθηκαν {len(numeric_labels)} label sequences")
    
    # Μετατροπή labels σε IOB format
    iob_labels = []
    total_tokens = 0
    
    for sentence_labels in numeric_labels:
        iob_tags = convert_numeric_to_iob_labels(sentence_labels, label_mapping)
        iob_labels.append(iob_tags)
        total_tokens += len(sentence_labels)
    
    # Δημιουργία νέου dataset
    iob_dataset = {
        "input": inputs,
        "label": iob_labels,
        "language": languages
    }
    
    # Αποθήκευση
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(iob_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ Αποθηκεύτηκε με {total_tokens} tokens")
    
    # Έλεγχος συνέπειας
    assert len(inputs) == len(iob_labels), "Mismatch σε αριθμό προτάσεων!"
    for i, (tokens, labels) in enumerate(zip(inputs, iob_labels)):
        assert len(tokens) == len(labels), f"Mismatch σε πρόταση {i}: {len(tokens)} tokens vs {len(labels)} labels"
    
    print(f"  ✅ Έλεγχος συνέπειας: OK")
    
    # Στατιστικά IOB labels
    all_iob_labels = [label for sentence in iob_labels for label in sentence]
    unique_labels = set(all_iob_labels)
    print(f"  📊 Unique IOB labels: {len(unique_labels)}")
    print(f"     {sorted(unique_labels)}")
    
    return len(unique_labels), total_tokens

def validate_iob_format(iob_labels):
    """
    Ελέγχει αν τα IOB labels είναι σωστά σχηματισμένα
    """
    errors = []
    
    for sentence_idx, sentence_labels in enumerate(iob_labels):
        for token_idx, label in enumerate(sentence_labels):
            if label.startswith('I-'):
                # Ελέγχει αν το I- ακολουθεί κατάλληλο B- ή I-
                if token_idx == 0:
                    errors.append(f"Sentence {sentence_idx}, Token {token_idx}: I- tag at start of sentence")
                else:
                    prev_label = sentence_labels[token_idx - 1]
                    entity_type = label[2:]  # Αφαιρεί το 'I-'
                    
                    if prev_label == 'O':
                        errors.append(f"Sentence {sentence_idx}, Token {token_idx}: I-{entity_type} after O")
                    elif prev_label.startswith('B-') or prev_label.startswith('I-'):
                        prev_entity_type = prev_label[2:]
                        if prev_entity_type != entity_type:
                            errors.append(f"Sentence {sentence_idx}, Token {token_idx}: I-{entity_type} after {prev_label}")
    
    return errors

# Paths - ΔΙΟΡΘΩΜΕΝΑ PATHS
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = script_dir  # Το script είναι ήδη στον GREEK_LEGAL_NER φάκελο
code_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))  # Πάμε στο CODE directory
mapping_file = os.path.join(code_dir, "greek_legal_ner_label_mapping.json")

# Datasets για μετατροπή
datasets = {
    "train": {
        "input": os.path.join(base_path, "train.json"),
        "output": os.path.join(base_path, "train_iob.json")
    },
    "test": {
        "input": os.path.join(base_path, "test.json"),
        "output": os.path.join(base_path, "test_iob.json")
    },
    "validation": {
        "input": os.path.join(base_path, "validation.json"),
        "output": os.path.join(base_path, "validation_iob.json")
    }
}

print("="*80)
print("ΜΕΤΑΤΡΟΠΗ DATASETS ΣΕ IOB FORMAT")
print("="*80)

# Φόρτωση mapping
fallback_mappings = [
    "greek_legal_ner_label_mapping.json",  # Σε current directory
    "../../greek_legal_ner_label_mapping.json",  # Δύο επίπεδα πάνω
    "../../../greek_legal_ner_label_mapping.json"  # Τρία επίπεδα πάνω
]

label_mapping = {}
for mapping_path in [mapping_file] + fallback_mappings:
    try:
        label_mapping = load_label_mapping(mapping_path)
        if label_mapping:
            print(f"✅ Βρέθηκε mapping στο: {mapping_path}")
            break
    except:
        continue

if not label_mapping:
    print("❌ Δεν βρέθηκε label mapping σε κανένα από τα paths!")
    for path in [mapping_file] + fallback_mappings:
        print(f"   Δοκιμάστηκε: {path}")
    exit(1)

print(f"✅ Φορτώθηκε mapping για {len(label_mapping)} labels")
print(f"   Mapping: {label_mapping}")

# Μετατροπή όλων των datasets
total_datasets = 0
total_tokens = 0
all_unique_labels = set()

for dataset_name, paths in datasets.items():
    if not os.path.exists(paths["input"]):
        print(f"⚠️  Το αρχείο {paths['input']} δεν υπάρχει! Παραλείπεται...")
        continue
    
    try:
        unique_labels_count, token_count = convert_dataset_to_iob(
            paths["input"], 
            paths["output"], 
            label_mapping
        )
        total_datasets += 1
        total_tokens += token_count
        
        # Συλλογή unique labels για validation
        with open(paths["output"], 'r', encoding='utf-8') as f:
            iob_data = json.load(f)
        
        dataset_labels = [label for sentence in iob_data["label"] for label in sentence]
        all_unique_labels.update(dataset_labels)
        
    except Exception as e:
        print(f"❌ Σφάλμα στη μετατροπή του {dataset_name}: {e}")

# Συνολικός έλεγχος
print("\n" + "="*80)
print("ΣΥΝΟΨΗ ΜΕΤΑΤΡΟΠΗΣ")
print("="*80)
print(f"✅ Μετατράπηκαν {total_datasets} datasets")
print(f"📊 Συνολικά tokens: {total_tokens:,}")
print(f"🏷️  Συνολικά unique IOB labels: {len(all_unique_labels)}")

print("\n📋 Όλα τα IOB labels:")
for label in sorted(all_unique_labels):
    print(f"   {label}")

# Έλεγχος IOB format correctness
print("\n🔍 ΈΛΕΓΧΟΣ IOB FORMAT CORRECTNESS:")
all_errors = []

for dataset_name, paths in datasets.items():
    if os.path.exists(paths["output"]):
        with open(paths["output"], 'r', encoding='utf-8') as f:
            iob_data = json.load(f)
        
        errors = validate_iob_format(iob_data["label"])
        if errors:
            print(f"❌ {dataset_name}: {len(errors)} errors found")
            all_errors.extend(errors)
        else:
            print(f"✅ {dataset_name}: No IOB format errors")

if all_errors:
    print(f"\n⚠️  Συνολικά βρέθηκαν {len(all_errors)} IOB format errors:")
    for error in all_errors[:10]:  # Show first 10 errors
        print(f"   {error}")
    if len(all_errors) > 10:
        print(f"   ... και {len(all_errors) - 10} περισσότερα")
else:
    print(f"\n✅ Όλα τα datasets έχουν σωστό IOB format!")

# Έλεγχος consistency με το ML model
print("\n🤖 ΈΛΕΓΧΟΣ CONSISTENCY ΜΕ ML MODEL:")

for dataset_name, paths in datasets.items():
    if os.path.exists(paths["output"]):
        with open(paths["output"], 'r', encoding='utf-8') as f:
            iob_data = json.load(f)
        
        # Έλεγχος δομής
        required_keys = ["input", "label", "language"]
        missing_keys = [key for key in required_keys if key not in iob_data]
        
        if missing_keys:
            print(f"❌ {dataset_name}: Λείπουν keys: {missing_keys}")
        else:
            # Έλεγχος μεγεθών
            input_len = len(iob_data["input"])
            label_len = len(iob_data["label"])
            lang_len = len(iob_data["language"])
            
            if input_len == label_len == lang_len:
                print(f"✅ {dataset_name}: Σωστή δομή ({input_len} προτάσεις)")
            else:
                print(f"❌ {dataset_name}: Mismatch - input:{input_len}, label:{label_len}, lang:{lang_len}")

print("="*80)
print("ΜΕΤΑΤΡΟΠΗ ΟΛΟΚΛΗΡΩΘΗΚΕ!")
print("="*80)
