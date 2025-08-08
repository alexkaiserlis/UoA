#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script για ΔΙΟΡΘΩΜΕΝΗ μετατροπή των αριθμητικών JSON datasets σε IOB format
Χρησιμοποιεί το σωστό mapping που βρέθηκε από την ανάλυση των HuggingFace datasets
"""

import json
import os

def get_correct_label_mapping():
    """
    Επιστρέφει το σωστό mapping από αριθμούς σε IOB labels
    Βασισμένο στην ανάλυση των HuggingFace datasets
    """
    return {
        0: 'O',
        1: 'B-ORG',
        2: 'I-ORG', 
        3: 'B-GPE',
        4: 'I-GPE',
        5: 'B-LEG-REFS',
        6: 'I-LEG-REFS',
        7: 'B-PUBLIC-DOCS',
        8: 'I-PUBLIC-DOCS',
        9: 'B-PERSON',
        10: 'I-PERSON',
        11: 'B-FACILITY',
        12: 'I-FACILITY',
        13: 'B-LOCATION-UNK',
        14: 'I-LOCATION-UNK',
        15: 'B-LOCATION-NAT',
        16: 'I-LOCATION-NAT'
    }

def convert_numeric_labels_to_correct_iob(labels_list):
    """
    Μετατρέπει αριθμητικά labels σε σωστά IOB strings
    
    Args:
        labels_list (list): Lista με τα αριθμητικά labels μιας πρότασης
        
    Returns:
        list: Lista με σωστά IOB tags
    """
    label_mapping = get_correct_label_mapping()
    iob_tags = []
    
    for label in labels_list:
        iob_tags.append(label_mapping.get(int(label), f'UNKNOWN_{int(label)}'))
    
    return iob_tags

def convert_dataset_to_correct_iob(input_file, output_file):
    """
    Μετατρέπει ένα dataset από αριθμητικά labels σε σωστό IOB format
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
        iob_tags = convert_numeric_labels_to_correct_iob(sentence_labels)
        iob_labels.append(iob_tags)
        total_tokens += len(sentence_labels)
    
    # Δημιουργία νέου dataset
    iob_dataset = {
        "input": inputs,
        "label": iob_labels,
        "language": languages
    }
    
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
    
    # Αποθήκευση
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(iob_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ Αποθηκεύτηκε με {total_tokens} tokens")
    
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

def main():
    """
    Κύρια συνάρτηση εκτέλεσης - διορθωμένη μετατροπή
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Datasets για μετατροπή
    datasets = {
        "train": {
            "input": os.path.join(script_dir, "train.json"),
            "output": os.path.join(script_dir, "train_iob_corrected.json")
        },
        "test": {
            "input": os.path.join(script_dir, "test.json"),
            "output": os.path.join(script_dir, "test_iob_corrected.json")
        },
        "validation": {
            "input": os.path.join(script_dir, "validation.json"),
            "output": os.path.join(script_dir, "validation_iob_corrected.json")
        }
    }

    print("="*80)
    print("ΔΙΟΡΘΩΜΕΝΗ ΜΕΤΑΤΡΟΠΗ DATASETS ΣΕ IOB FORMAT")
    print("="*80)

    # Εμφάνιση του σωστού mapping
    mapping = get_correct_label_mapping()
    print(f"✅ Χρησιμοποιώ το σωστό mapping:")
    for num, label in sorted(mapping.items()):
        print(f"   {num}: {label}")
    print()

    # Μετατροπή όλων των datasets
    total_datasets = 0
    total_tokens = 0
    all_unique_labels = set()

    for dataset_name, paths in datasets.items():
        if not os.path.exists(paths["input"]):
            print(f"⚠️  Το αρχείο {paths['input']} δεν υπάρχει! Παραλείπεται...")
            continue
        
        try:
            unique_labels_count, token_count = convert_dataset_to_correct_iob(
                paths["input"], 
                paths["output"]
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
    print("ΣΥΝΟΨΗ ΔΙΟΡΘΩΜΕΝΗΣ ΜΕΤΑΤΡΟΠΗΣ")
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

    print("="*80)
    print("ΔΙΟΡΘΩΜΕΝΗ ΜΕΤΑΤΡΟΠΗ ΟΛΟΚΛΗΡΩΘΗΚΕ!")
    print("="*80)

if __name__ == "__main__":
    main()
