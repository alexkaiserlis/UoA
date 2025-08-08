#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ανάλυση μοναδικών NER tags στο train set
"""

from datasets import load_dataset
from collections import Counter

def analyze_train_ner_tags():
    """
    Αναλύει τις μοναδικές τιμές NER tags στο train set
    """
    print("🔍 ΑΝΑΛΥΣΗ ΜΟΝΑΔΙΚΩΝ NER TAGS ΣΤΟ TRAIN SET")
    print("="*55)
    
    # Φόρτωση dataset
    print("Φόρτωση dataset...")
    ds = load_dataset("joelniklaus/greek_legal_ner")
    
    # Συλλογή όλων των tags από το train set
    unique_tags = set()
    tag_counter = Counter()
    total_tokens = 0
    total_docs = 0
    
    print("Ανάλυση train set...")
    
    for example in ds['train']:
        total_docs += 1
        ner_tags = example['ner']
        total_tokens += len(ner_tags)
        
        # Συλλογή μοναδικών tags και μέτρηση
        for tag in ner_tags:
            unique_tags.add(tag)
            tag_counter[tag] += 1
    
    print(f"   📄 Συνολικά έγγραφα: {total_docs:,}")
    print(f"   🔤 Συνολικά tokens: {total_tokens:,}")
    print(f"   🏷️  Μοναδικά NER tags: {len(unique_tags)}")
    
    print("\n" + "="*60)
    print("📋 ΟΛΕΣ ΟΙ ΜΟΝΑΔΙΚΕΣ ΤΙΜΕΣ NER TAGS")
    print("="*60)
    
    # Ταξινόμηση και εκτύπωση όλων των μοναδικών tags
    sorted_unique_tags = sorted(unique_tags)
    
    print("Λίστα όλων των μοναδικών τιμών:")
    print("-" * 35)
    for i, tag in enumerate(sorted_unique_tags, 1):
        count = tag_counter[tag]
        percentage = (count / total_tokens) * 100
        print(f"   {i:2d}. {tag:<20} ({count:>8,} φορές, {percentage:>5.2f}%)")
    
    print(f"\n📊 ΣΥΓΚΕΝΤΡΩΤΙΚΗ ΑΝΑΛΥΣΗ:")
    print("-" * 30)
    print(f"Συνολικές μοναδικές τιμές: {len(unique_tags)}")
    
    # Ανάλυση ανά τύπο
    o_tags = [tag for tag in unique_tags if tag == 'O']
    b_tags = [tag for tag in unique_tags if tag.startswith('B-')]
    i_tags = [tag for tag in unique_tags if tag.startswith('I-')]
    
    print(f"O tags: {len(o_tags)} τύπος")
    print(f"B- tags: {len(b_tags)} τύποι")
    print(f"I- tags: {len(i_tags)} τύποι")
    
    print(f"\n🏷️  ΑΝΑΛΥΤΙΚΗ ΚΑΤΗΓΟΡΙΟΠΟΙΗΣΗ:")
    print("-" * 40)
    
    print("O TAGS:")
    for tag in o_tags:
        count = tag_counter[tag]
        print(f"   {tag}: {count:,} φορές")
    
    print(f"\nB- TAGS (Beginning of Entity):")
    for tag in sorted(b_tags):
        count = tag_counter[tag]
        print(f"   {tag}: {count:,} φορές")
    
    print(f"\nI- TAGS (Inside Entity):")
    for tag in sorted(i_tags):
        count = tag_counter[tag]
        print(f"   {tag}: {count:,} φορές")
    
    # Ανάλυση entity types
    entity_types = set()
    for tag in unique_tags:
        if tag.startswith('B-') or tag.startswith('I-'):
            entity_type = tag[2:]  # Αφαίρεση B- ή I-
            entity_types.add(entity_type)
    
    print(f"\n🎯 ENTITY TYPES ΠΟΥ ΒΡΕΘΗΚΑΝ:")
    print("-" * 35)
    print(f"Συνολικά entity types: {len(entity_types)}")
    
    for i, entity_type in enumerate(sorted(entity_types), 1):
        b_tag = f"B-{entity_type}"
        i_tag = f"I-{entity_type}"
        b_count = tag_counter.get(b_tag, 0)
        i_count = tag_counter.get(i_tag, 0)
        total_count = b_count + i_count
        
        print(f"   {i:2d}. {entity_type}")
        print(f"       B-{entity_type}: {b_count:,} φορές")
        print(f"       I-{entity_type}: {i_count:,} φορές")
        print(f"       Σύνολο: {total_count:,} φορές")
        print()
    
    return {
        'unique_tags': sorted_unique_tags,
        'tag_counts': dict(tag_counter),
        'entity_types': sorted(entity_types),
        'total_unique': len(unique_tags)
    }

if __name__ == "__main__":
    results = analyze_train_ner_tags()
    print(f"✅ Ανάλυση ολοκληρώθηκε! Βρέθηκαν {results['total_unique']} μοναδικές τιμές NER tags.")
