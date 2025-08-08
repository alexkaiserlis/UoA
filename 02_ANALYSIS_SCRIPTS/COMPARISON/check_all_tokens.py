#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Έλεγχος αν η διαφορά οφείλεται στο ότι το README μετράει B- + I- tags
"""

import json
import os
from collections import Counter

def count_all_entity_tokens():
    """
    Μετράει όλα τα entity tokens (B- + I-) αντί μόνο τα B- (entities)
    """
    dataset_folder = r"C:\Users\User\Το Drive μου\AEGEAN UNIVERSITY\LEGAL DOCUMENTS ARCHIVE\ΠΑΙΓΑΙΟΥ\CODE\NER MODEL\GREEKLEGALNER DATASET"
    
    splits = {
        'train': 'train.jsonl',
        'validation': 'validation.jsonl', 
        'test': 'test.jsonl'
    }
    
    results = {}
    
    print("🔍 ΜΕΤΡΗΣΗ ΟΛΩΝ ΤΩΝ ENTITY TOKENS (B- + I- tags)")
    print("="*80)
    
    for split_name, filename in splits.items():
        file_path = os.path.join(dataset_folder, filename)
        
        if not os.path.exists(file_path):
            continue
        
        # Counters για B- και I- tags
        entity_token_counts = Counter()
        
        # Διάβασμα δεδομένων
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    example = json.loads(line)
                    
                    if 'ner' in example:
                        for tag in example['ner']:
                            if tag.startswith('B-') or tag.startswith('I-'):
                                entity_type = tag[2:]  # Αφαίρεση B- ή I-
                                entity_token_counts[entity_type] += 1
        
        results[split_name] = entity_token_counts
        
        total_entity_tokens = sum(entity_token_counts.values())
        print(f"\n📁 {split_name.upper()} split:")
        print(f"  Σύνολο entity tokens (B- + I-): {total_entity_tokens:,}")
    
    return results

def compare_with_readme_all_tokens(our_results):
    """
    Σύγκριση με τα επίσημα στατιστικά χρησιμοποιώντας όλα τα entity tokens
    """
    # Επίσημα στατιστικά από το README
    official_stats = {
        'FACILITY': {'train': 1224, 'validation': 60, 'test': 142},
        'GPE': {'train': 5400, 'validation': 1214, 'test': 1083},
        'LEG-REFS': {'train': 5159, 'validation': 1382, 'test': 1331},
        'LOCATION-NAT': {'train': 145, 'validation': 2, 'test': 26},
        'LOCATION-UNK': {'train': 1316, 'validation': 283, 'test': 205},
        'ORG': {'train': 5906, 'validation': 1506, 'test': 1354},
        'PERSON': {'train': 1921, 'validation': 475, 'test': 491},
        'PUBLIC-DOCS': {'train': 2652, 'validation': 556, 'test': 452}
    }
    
    print(f"\n" + "="*100)
    print("📊 ΣΥΓΚΡΙΣΗ ΜΕ ΕΠΙΣΗΜΑ ΣΤΑΤΙΣΤΙΚΑ (Όλα τα entity tokens)")
    print("="*100)
    
    print(f"{'Κατηγορία':<15} {'Split':<10} {'README':<10} {'B-+I- tags':<12} {'Διαφορά':<10} {'Ποσοστό':<10}")
    print("-" * 100)
    
    total_official = 0
    total_ours = 0
    
    for category in sorted(official_stats.keys()):
        for split in ['train', 'validation', 'test']:
            official_count = official_stats[category][split]
            our_count = our_results.get(split, {}).get(category, 0)
            
            difference = our_count - official_count
            percentage = (our_count / official_count * 100) if official_count > 0 else 0
            
            total_official += official_count
            total_ours += our_count
            
            print(f"{category:<15} {split:<10} {official_count:<10} {our_count:<12} {difference:<10} {percentage:<10.1f}%")
    
    print("-" * 100)
    total_diff = total_ours - total_official
    total_percentage = (total_ours / total_official * 100) if total_official > 0 else 0
    print(f"{'ΣΥΝΟΛΟ':<15} {'ALL':<10} {total_official:<10} {total_ours:<12} {total_diff:<10} {total_percentage:<10.1f}%")
    
    print(f"\n📊 ΑΠΟΤΕΛΕΣΜΑΤΑ:")
    print(f"• Επίσημα: {total_official:,}")
    print(f"• Δικά μας (B- + I-): {total_ours:,}")
    print(f"• Ακρίβεια: {total_percentage:.1f}%")
    
    if abs(total_percentage - 100) < 5:
        print("✅ Πολύ κοντά! Η διαφορά πιθανώς οφείλεται σε μικρές αλλαγές στο dataset.")
    elif total_percentage > 95:
        print("✅ Σχεδόν ταυτίζονται! Μικρές διαφορές είναι αναμενόμενες.")
    else:
        print("⚠️  Εξακολουθούν να υπάρχουν διαφορές. Πιθανώς διαφορετική έκδοση dataset.")

if __name__ == "__main__":
    results = count_all_entity_tokens()
    compare_with_readme_all_tokens(results)
