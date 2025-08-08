#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Απλός κώδικας για μέτρηση I και B tags ξεχωριστά στο train set
"""

import json
from collections import Counter
from pathlib import Path

def count_i_b_tags_separately():
    """
    Μετράει τα I και B tags ξεχωριστά στο train set
    """
    # Το train.json είναι στο ../../../../01_DATASETS/GREEK_LEGAL_NER/NER MODEL/
    train_file = Path("../../../../01_DATASETS/GREEK_LEGAL_NER/NER MODEL/train.json")
    
    if not train_file.exists():
        print(f"❌ Το αρχείο {train_file} δεν βρέθηκε!")
        return
    
    print("🔍 ΜΕΤΡΗΣΗ I ΚΑΙ B TAGS ΞΕΧΩΡΙΣΤΑ ΣΤΟ TRAIN SET")
    print("="*56)
    
    # Counters για I και B tags ξεχωριστά
    b_tags_counter = Counter()
    i_tags_counter = Counter()
    total_tokens = 0
    total_documents = 0
    
    # Διάβασμα train.json - είναι JSON αρχείο με δομή HuggingFace dataset
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Το train.json έχει δομή: {"features": [...], "rows": [...]}
    rows = data.get('rows', [])
    
    for row_data in rows:
        total_documents += 1
        
        # Παίρνουμε τα δεδομένα από το row
        row_content = row_data.get('row', {})
        input_tokens = row_content.get('input', [])
        labels = row_content.get('label', [])
        
        # Το mapping των labels βρίσκεται στα features
        features = data.get('features', [])
        label_feature = None
        for feature in features:
            if feature.get('name') == 'label':
                label_feature = feature
                break
        
        if label_feature and 'names' in label_feature['type']['feature']:
            label_names = label_feature['type']['feature']['names']
            
            # Μετατρέπουμε τα numeric labels σε string labels
            string_labels = []
            for label_idx in labels:
                if 0 <= label_idx < len(label_names):
                    string_labels.append(label_names[label_idx])
                else:
                    string_labels.append("UNKNOWN")
            
            total_tokens += len(string_labels)
            
            # Μετράμε κάθε tag
            for tag in string_labels:
                if tag.startswith('B-'):
                    # Αφαιρούμε το B- για να πάρουμε την κατηγορία
                    category = tag[2:]  # π.χ. B-ORG -> ORG
                    b_tags_counter[category] += 1
                elif tag.startswith('I-'):
                    # Αφαιρούμε το I- για να πάρουμε την κατηγορία
                    category = tag[2:]  # π.χ. I-ORG -> ORG
                    i_tags_counter[category] += 1
    
    print(f"📊 ΣΤΑΤΙΣΤΙΚΑ TRAIN SET:")
    print(f"   📄 Συνολικά έγγραφα: {total_documents:,}")
    print(f"   🔤 Συνολικά tokens: {total_tokens:,}")
    
    print(f"\n🅱️  B-TAGS (Beginning of Entity):")
    print("-" * 40)
    total_b_tags = 0
    for category, count in sorted(b_tags_counter.items()):
        print(f"   B-{category:<15}: {count:>6,}")
        total_b_tags += count
    print(f"   {'ΣΥΝΟΛΟ B-TAGS':<18}: {total_b_tags:>6,}")
    
    print(f"\n🅸  I-TAGS (Inside Entity):")
    print("-" * 40)
    total_i_tags = 0
    for category, count in sorted(i_tags_counter.items()):
        print(f"   I-{category:<15}: {count:>6,}")
        total_i_tags += count
    print(f"   {'ΣΥΝΟΛΟ I-TAGS':<18}: {total_i_tags:>6,}")
    
    print(f"\n📈 ΣΥΓΚΕΝΤΡΩΤΙΚΑ:")
    print("-" * 30)
    print(f"   B-tags: {total_b_tags:,}")
    print(f"   I-tags: {total_i_tags:,}")
    print(f"   Σύνολο Entity Tags: {total_b_tags + total_i_tags:,}")
    
    # Αναλυτικός πίνακας ανά κατηγορία
    print(f"\n📋 ΑΝΑΛΥΤΙΚΟΣ ΠΙΝΑΚΑΣ ΑΝΑ ΚΑΤΗΓΟΡΙΑ:")
    print("-" * 60)
    print(f"{'Κατηγορία':<15} {'B-tags':<10} {'I-tags':<10} {'Σύνολο':<10}")
    print("-" * 60)
    
    all_categories = set(list(b_tags_counter.keys()) + list(i_tags_counter.keys()))
    category_totals = {}
    
    for category in sorted(all_categories):
        b_count = b_tags_counter.get(category, 0)
        i_count = i_tags_counter.get(category, 0)
        total = b_count + i_count
        category_totals[category] = total
        print(f"{category:<15} {b_count:<10,} {i_count:<10,} {total:<10,}")
    
    print("-" * 60)
    print(f"{'ΣΥΝΟΛΟ':<15} {total_b_tags:<10,} {total_i_tags:<10,} {total_b_tags + total_i_tags:<10,}")
    
    # Ποσοστά
    print(f"\n📊 ΠΟΣΟΣΤΑ:")
    print("-" * 40)
    for category in sorted(all_categories):
        percentage = (category_totals[category] / (total_b_tags + total_i_tags)) * 100
        print(f"   {category:<15}: {percentage:>5.1f}%")
    
    return {
        'b_tags': dict(b_tags_counter),
        'i_tags': dict(i_tags_counter),
        'total_b': total_b_tags,
        'total_i': total_i_tags,
        'total_documents': total_documents,
        'total_tokens': total_tokens
    }

if __name__ == "__main__":
    results = count_i_b_tags_separately()
