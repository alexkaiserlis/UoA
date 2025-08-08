#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ανάλυση όλων των splits (train, test, validation) για IOB tags
Μετράει ποια και πόσα tags έχει κάθε split και πόσα entities συνολικά
"""

import json
from collections import Counter
from pathlib import Path

def analyze_split(split_name, file_path):
    """
    Αναλύει ένα split και επιστρέφει στατιστικά
    """
    if not file_path.exists():
        print(f"❌ Το αρχείο {file_path} δεν βρέθηκε!")
        return None
    
    print(f"🔍 ΑΝΑΛΥΣΗ {split_name.upper()} SET")
    print("="*60)
    
    # Counters
    b_tags_counter = Counter()
    i_tags_counter = Counter()
    o_tags_counter = 0
    total_tokens = 0
    total_sentences = 0
    
    # Διάβασμα αρχείου
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Το αρχείο έχει δομή: {"input": [...], "label": [...], "language": [...]}
    labels_data = data.get('label', [])
    input_data = data.get('input', [])
    
    total_sentences = len(labels_data)
    
    for sentence_labels in labels_data:
        total_tokens += len(sentence_labels)
        
        # Μετράμε κάθε tag
        for tag in sentence_labels:
            if tag == 'O':
                o_tags_counter += 1
            elif tag.startswith('B-'):
                # Αφαιρούμε το B- για να πάρουμε την κατηγορία
                category = tag[2:]  # π.χ. B-ORG -> ORG
                b_tags_counter[category] += 1
            elif tag.startswith('I-'):
                # Αφαιρούμε το I- για να πάρουμε την κατηγορία
                category = tag[2:]  # π.χ. I-ORG -> ORG
                i_tags_counter[category] += 1
    
    print(f"📊 ΣΤΑΤΙΣΤΙΚΑ {split_name.upper()} SET:")
    print(f"   📄 Συνολικές προτάσεις: {total_sentences:,}")
    print(f"   🔤 Συνολικά tokens: {total_tokens:,}")
    print(f"   ⭕ O tags: {o_tags_counter:,}")
    
    print(f"\n🅱️  B-TAGS (Beginning of Entity):")
    print("-" * 40)
    total_b_tags = 0
    for category, count in sorted(b_tags_counter.items()):
        print(f"   B-{category:<20}: {count:>8,}")
        total_b_tags += count
    print(f"   {'ΣΥΝΟΛΟ B-TAGS':<23}: {total_b_tags:>8,}")
    
    print(f"\n🅸  I-TAGS (Inside Entity):")
    print("-" * 40)
    total_i_tags = 0
    for category, count in sorted(i_tags_counter.items()):
        print(f"   I-{category:<20}: {count:>8,}")
        total_i_tags += count
    print(f"   {'ΣΥΝΟΛΟ I-TAGS':<23}: {total_i_tags:>8,}")
    
    print(f"\n📈 ΣΥΓΚΕΝΤΡΩΤΙΚΑ:")
    print("-" * 30)
    print(f"   O tags: {o_tags_counter:,}")
    print(f"   B-tags: {total_b_tags:,}")
    print(f"   I-tags: {total_i_tags:,}")
    print(f"   Σύνολο Entity Tags: {total_b_tags + total_i_tags:,}")
    print(f"   Σύνολο όλων των tags: {total_tokens:,}")
    
    # Αναλυτικός πίνακας ανά κατηγορία
    print(f"\n📋 ΑΝΑΛΥΤΙΚΟΣ ΠΙΝΑΚΑΣ ΑΝΑ ΚΑΤΗΓΟΡΙΑ:")
    print("-" * 70)
    print(f"{'Κατηγορία':<20} {'B-tags':<12} {'I-tags':<12} {'Σύνολο':<12}")
    print("-" * 70)
    
    all_categories = set(list(b_tags_counter.keys()) + list(i_tags_counter.keys()))
    category_totals = {}
    
    for category in sorted(all_categories):
        b_count = b_tags_counter.get(category, 0)
        i_count = i_tags_counter.get(category, 0)
        total = b_count + i_count
        category_totals[category] = total
        print(f"{category:<20} {b_count:<12,} {i_count:<12,} {total:<12,}")
    
    print("-" * 70)
    print(f"{'ΣΥΝΟΛΟ':<20} {total_b_tags:<12,} {total_i_tags:<12,} {total_b_tags + total_i_tags:<12,}")
    
    # Ποσοστά
    total_entity_tags = total_b_tags + total_i_tags
    if total_entity_tags > 0:
        print(f"\n📊 ΠΟΣΟΣΤΑ ΚΑΤΗΓΟΡΙΩΝ:")
        print("-" * 50)
        for category in sorted(all_categories):
            percentage = (category_totals[category] / total_entity_tags) * 100
            print(f"   {category:<20}: {percentage:>6.1f}%")
    
    print("\n" + "="*60 + "\n")
    
    return {
        'split_name': split_name,
        'b_tags': dict(b_tags_counter),
        'i_tags': dict(i_tags_counter),
        'o_tags': o_tags_counter,
        'total_b': total_b_tags,
        'total_i': total_i_tags,
        'total_entities': total_b_tags + total_i_tags,
        'total_sentences': total_sentences,
        'total_tokens': total_tokens,
        'categories': sorted(all_categories)
    }

def compare_splits(results):
    """
    Συγκρίνει τα αποτελέσματα από όλα τα splits
    """
    print("🔄 ΣΥΓΚΡΙΤΙΚΗ ΑΝΑΛΥΣΗ SPLITS")
    print("="*80)
    
    # Συγκεντρωτικός πίνακας
    print(f"\n📊 ΣΥΓΚΕΝΤΡΩΤΙΚΟΣ ΠΙΝΑΚΑΣ:")
    print("-" * 80)
    print(f"{'Split':<12} {'Προτάσεις':<12} {'Tokens':<12} {'O-tags':<12} {'B-tags':<12} {'I-tags':<12} {'Entities':<12}")
    print("-" * 80)
    
    total_sentences = 0
    total_tokens = 0
    total_o = 0
    total_b = 0
    total_i = 0
    total_entities = 0
    
    for result in results:
        if result:
            print(f"{result['split_name']:<12} {result['total_sentences']:<12,} {result['total_tokens']:<12,} "
                  f"{result['o_tags']:<12,} {result['total_b']:<12,} {result['total_i']:<12,} {result['total_entities']:<12,}")
            
            total_sentences += result['total_sentences']
            total_tokens += result['total_tokens']
            total_o += result['o_tags']
            total_b += result['total_b']
            total_i += result['total_i']
            total_entities += result['total_entities']
    
    print("-" * 80)
    print(f"{'ΣΥΝΟΛΟ':<12} {total_sentences:<12,} {total_tokens:<12,} {total_o:<12,} {total_b:<12,} {total_i:<12,} {total_entities:<12,}")
    
    # Ποσοστά ανά split
    print(f"\n📈 ΠΟΣΟΣΤΑ ΑΝΑ SPLIT:")
    print("-" * 60)
    print(f"{'Split':<12} {'% Προτάσεων':<15} {'% Tokens':<12} {'% Entities':<12}")
    print("-" * 60)
    
    for result in results:
        if result:
            sent_pct = (result['total_sentences'] / total_sentences) * 100 if total_sentences > 0 else 0
            token_pct = (result['total_tokens'] / total_tokens) * 100 if total_tokens > 0 else 0
            entity_pct = (result['total_entities'] / total_entities) * 100 if total_entities > 0 else 0
            
            print(f"{result['split_name']:<12} {sent_pct:<15.1f} {token_pct:<12.1f} {entity_pct:<12.1f}")
    
    # Ανάλυση κατηγοριών ανά split
    all_categories = set()
    for result in results:
        if result:
            all_categories.update(result['categories'])
    
    print(f"\n📋 ΚΑΤΗΓΟΡΙΕΣ ΑΝΑ SPLIT:")
    print("-" * 100)
    header = f"{'Κατηγορία':<20}"
    for result in results:
        if result:
            header += f" {result['split_name']:<15}"
    print(header)
    print("-" * 100)
    
    for category in sorted(all_categories):
        row = f"{category:<20}"
        for result in results:
            if result:
                b_count = result['b_tags'].get(category, 0)
                i_count = result['i_tags'].get(category, 0)
                total_cat = b_count + i_count
                row += f" {total_cat:<15,}"
        print(row)

def main():
    """
    Κύρια συνάρτηση - αναλύει όλα τα splits
    """
    # Paths για τα IOB αρχεία
    base_path = Path("../../../01_DATASETS/GREEK_LEGAL_NER")
    
    splits = [
        ("train", base_path / "train_iob.json"),
        ("test", base_path / "test_iob.json"),
        ("validation", base_path / "validation_iob.json")
    ]
    
    results = []
    
    # Ανάλυση κάθε split
    for split_name, file_path in splits:
        result = analyze_split(split_name, file_path)
        results.append(result)
    
    # Συγκριτική ανάλυση
    compare_splits(results)
    
    return results

if __name__ == "__main__":
    results = main()
