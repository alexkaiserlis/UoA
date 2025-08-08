#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script για εξαγωγή και έλεγχο των πραγματικών ground truth entities από το Greek Legal NER dataset
ΔΙΟΡΘΩΜΕΝΗ ΕΚΔΟΣΗ - Λύνει το πρόβλημα με τους λάθος αριθμούς entities
"""

import json
from datasets import load_dataset
from collections import defaultdict, Counter
import pandas as pd

def extract_entities_from_bio_corrected(words, labels):
    """
    ΔΙΟΡΘΩΜΕΝΗ εξαγωγή entities από BIO format
    Χειρίζεται σωστά τα B- και I- tags
    """
    entities = []
    current_entity = None
    current_entity_words = []
    start_idx = 0
    
    for i, (word, label) in enumerate(zip(words, labels)):
        if label.startswith('B-'):
            # Αν υπάρχει ανολοκλήρωτο entity, το προσθέτουμε πρώτα
            if current_entity is not None:
                entities.append({
                    'text': ' '.join(current_entity_words),
                    'label': current_entity,
                    'start_word': start_idx,
                    'end_word': i - 1,
                    'words': current_entity_words.copy(),
                    'length': len(current_entity_words)
                })
            
            # Ξεκινάμε νέο entity
            current_entity = label[2:]  # Αφαιρούμε το 'B-'
            current_entity_words = [word]
            start_idx = i
            
        elif label.startswith('I-'):
            # Συνεχίζουμε το τρέχον entity ΜΟΝΟ αν είναι το ίδιο type
            entity_type = label[2:]  # Αφαιρούμε το 'I-'
            if current_entity is not None and entity_type == current_entity:
                current_entity_words.append(word)
            else:
                # Λάθος I- tag χωρίς προηγούμενο B- ή διαφορετικό type
                # Τελειώνουμε το προηγούμενο entity (αν υπάρχει)
                if current_entity is not None:
                    entities.append({
                        'text': ' '.join(current_entity_words),
                        'label': current_entity,
                        'start_word': start_idx,
                        'end_word': i - 1,
                        'words': current_entity_words.copy(),
                        'length': len(current_entity_words)
                    })
                
                # Αρχίζουμε νέο entity από αυτό το I- tag (θεωρούμε ότι είναι B-)
                current_entity = entity_type
                current_entity_words = [word]
                start_idx = i
                
        else:
            # 'O' label ή άλλη περίπτωση - τελειώνουμε το τρέχον entity
            if current_entity is not None:
                entities.append({
                    'text': ' '.join(current_entity_words),
                    'label': current_entity,
                    'start_word': start_idx,
                    'end_word': i - 1,
                    'words': current_entity_words.copy(),
                    'length': len(current_entity_words)
                })
                current_entity = None
                current_entity_words = []
    
    # Αν το document τελειώνει με entity
    if current_entity is not None:
        entities.append({
            'text': ' '.join(current_entity_words),
            'label': current_entity,
            'start_word': start_idx,
            'end_word': len(words) - 1,
            'words': current_entity_words.copy(),
            'length': len(current_entity_words)
        })
    
    return entities

def verify_dataset_counts():
    """
    Επαληθεύει τους πραγματικούς αριθμούς entities στο dataset
    Συγκρίνει με τα δεδομένα που έδωσες
    """
    print("🔍 ΕΠΑΛΗΘΕΥΣΗ DATASET COUNTS...")
    ds = load_dataset("joelniklaus/greek_legal_ner")
    
    # Καταμέτρηση με δύο μεθόδους για έλεγχο
    method1_counts = defaultdict(lambda: defaultdict(int))  # Μέτρηση tokens
    method2_counts = defaultdict(lambda: defaultdict(int))  # Μέτρηση entities
    
    for split in ['train', 'validation', 'test']:
        print(f"\n📊 Ανάλυση split: {split}")
        split_data = ds[split]
        
        # Method 1: Μέτρηση tokens (το παλιό μας)
        for example in split_data:
            labels = example['ner']
            for label in labels:
                if label != 'O':
                    method1_counts[label][split] += 1
        
        # Method 2: Μέτρηση entities (διορθωμένο)
        for example in split_data:
            words = example['words']
            labels = example['ner']
            entities = extract_entities_from_bio_corrected(words, labels)
            for entity in entities:
                method2_counts[entity['label']][split] += 1
    
    # Σύγκριση με τα δεδομένα που έδωσες
    expected_counts = {
        ('FACILITY', 'test'): 142,
        ('FACILITY', 'train'): 1224,
        ('FACILITY', 'validation'): 60,
        ('GPE', 'test'): 1083,
        ('GPE', 'train'): 5400,
        ('GPE', 'validation'): 1214,
        ('LEG-REFS', 'test'): 1331,
        ('LEG-REFS', 'train'): 5159,
        ('LEG-REFS', 'validation'): 1382,
        ('LOCATION-NAT', 'test'): 26,
        ('LOCATION-NAT', 'train'): 145,
        ('LOCATION-NAT', 'validation'): 2,
        ('LOCATION-UNK', 'test'): 205,
        ('LOCATION-UNK', 'train'): 1316,
        ('LOCATION-UNK', 'validation'): 283,
        ('ORG', 'test'): 1354,
        ('ORG', 'train'): 5906,
        ('ORG', 'validation'): 1506,
        ('PERSON', 'test'): 491,
        ('PERSON', 'train'): 1921,
        ('PERSON', 'validation'): 475,
        ('PUBLIC-DOCS', 'test'): 452,
        ('PUBLIC-DOCS', 'train'): 2652,
        ('PUBLIC-DOCS', 'validation'): 556,
    }
    
    print(f"\n{'='*80}")
    print("📊 ΣΥΓΚΡΙΣΗ ΜΕΘΟΔΩΝ ΚΑΤΑΜΕΤΡΗΣΗΣ")
    print(f"{'='*80}")
    print(f"{'Entity + Split':<25} {'Expected':<10} {'Method1(Tokens)':<15} {'Method2(Entities)':<15} {'Match?'}")
    print("-" * 80)
    
    all_entity_types = set()
    for split in ['train', 'validation', 'test']:
        all_entity_types.update(method1_counts.keys())
        all_entity_types.update(method2_counts.keys())
    
    total_mismatches = 0
    correct_method = None
    
    for entity_type in sorted(all_entity_types):
        for split in ['test', 'train', 'validation']:
            key = (entity_type, split)
            expected = expected_counts.get(key, 0)
            method1_count = method1_counts[entity_type][split]
            method2_count = method2_counts[entity_type][split]
            
            # Έλεγχος ποια μέθοδος είναι σωστή
            method1_match = "✅" if method1_count == expected else "❌"
            method2_match = "✅" if method2_count == expected else "❌"
            
            if method1_count == expected:
                correct_method = "Method1 (Tokens)"
            elif method2_count == expected:
                correct_method = "Method2 (Entities)"
            
            if method1_count != expected and method2_count != expected:
                total_mismatches += 1
            
            print(f"{key[0] + '+' + key[1]:<25} {expected:<10} {method1_count:<15} {method2_count:<15} {method1_match}/{method2_match}")
    
    print(f"\n🎯 ΑΠΟΤΕΛΕΣΜΑΤΑ:")
    print(f"   Συνολικές ασυμφωνίες: {total_mismatches}")
    if correct_method:
        print(f"   Σωστή μέθοδος: {correct_method}")
    else:
        print(f"   ⚠️  Καμία μέθοδος δεν ταιριάζει πλήρως με τα expected counts!")
    
    return method1_counts, method2_counts, expected_counts

def main():
    print("🔍 ΔΙΟΡΘΩΜΕΝΗ ΕΚΔΟΣΗ - Επαλήθευση Ground Truth Entities")
    print("="*70)
    
    # Επαλήθευση counts
    method1_counts, method2_counts, expected_counts = verify_dataset_counts()
    
    # Φόρτωση dataset για πλήρη ανάλυση
    print("\n📁 Φόρτωση dataset για πλήρη ανάλυση...")
    ds = load_dataset("joelniklaus/greek_legal_ner")
    test_set = ds['test']
    
    print(f"Test set έχει {len(test_set)} documents")
    
    # Εξαγωγή entities με τη διορθωμένη μέθοδο
    all_entities = []
    entity_stats = defaultdict(lambda: defaultdict(int))
    document_stats = []
    
    print("Εξαγωγή entities από κάθε έγγραφο...")
    
    for doc_id, example in enumerate(test_set):
        words = example['words']
        labels = example['ner']
        
        # Εξαγωγή entities με διορθωμένη μέθοδο
        entities = extract_entities_from_bio_corrected(words, labels)
        
        # Στατιστικά για το document
        doc_stats = {
            'doc_id': doc_id,
            'total_words': len(words),
            'total_entities': len(entities),
            'entities_by_type': Counter([e['label'] for e in entities])
        }
        document_stats.append(doc_stats)
        
        # Προσθήκη του doc_id σε κάθε entity
        for entity in entities:
            entity['doc_id'] = doc_id
            entity['doc_total_words'] = len(words)
            all_entities.append(entity)
            
            # Ενημέρωση στατιστικών
            entity_type = entity['label']
            entity_length = len(entity['words'])
            entity_stats[entity_type]['count'] += 1
            entity_stats[entity_type]['total_words'] += entity_length
            entity_stats[entity_type]['min_words'] = min(
                entity_stats[entity_type].get('min_words', float('inf')), 
                entity_length
            )
            entity_stats[entity_type]['max_words'] = max(
                entity_stats[entity_type].get('max_words', 0), 
                entity_length
            )
    
    # Σύγκριση των αποτελεσμάτων με τα expected
    print(f"\n🎯 ΤΕΛΙΚΗ ΣΥΓΚΡΙΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ:")
    print("-" * 50)
    total_entities_found = len(all_entities)
    entity_counts_by_type = Counter([e['label'] for e in all_entities])
    
    for entity_type in sorted(entity_counts_by_type.keys()):
        found_count = entity_counts_by_type[entity_type]
        expected_count = expected_counts.get((entity_type, 'test'), 0)
        match_status = "✅" if found_count == expected_count else "❌"
        print(f"{entity_type:<15}: Found {found_count:>4}, Expected {expected_count:>4} {match_status}")
    
    print(f"\nΣυνολικά entities: {total_entities_found}")
    
    # Αποθήκευση διορθωμένων αποτελεσμάτων
    print("\nΑποθήκευση διορθωμένων αποτελεσμάτων...")
    
    with open('corrected_groundtruth_entities.json', 'w', encoding='utf-8') as f:
        json.dump(all_entities, f, ensure_ascii=False, indent=2)
    
    with open('corrected_entity_counts_comparison.json', 'w', encoding='utf-8') as f:
        comparison_data = {
            'found_counts': dict(entity_counts_by_type),
            'expected_counts': {k[0]: v for k, v in expected_counts.items() if k[1] == 'test'},
            'method1_counts': {k: v['test'] for k, v in method1_counts.items()},
            'method2_counts': {k: v['test'] for k, v in method2_counts.items()},
            'total_entities_found': total_entities_found
        }
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Διορθωμένα αποτελέσματα αποθηκεύτηκαν!")

if __name__ == "__main__":
    main()
