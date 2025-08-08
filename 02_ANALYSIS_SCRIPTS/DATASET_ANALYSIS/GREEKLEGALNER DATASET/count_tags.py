#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Απλός κώδικας για μέτρηση όλων των NER tags στο dataset
"""

from datasets import load_dataset
from collections import Counter

def count_all_ner_tags():
    """
    Μετράει όλα τα διαφορετικά NER tags στο train set μόνο
    """
    print("🔍 ΜΕΤΡΗΣΗ ΟΛΩΝ ΤΩΝ NER TAGS ΣΤΟ TRAIN SET")
    print("="*50)
    
    # Φόρτωση dataset
    print("Φόρτωση dataset...")
    ds = load_dataset("joelniklaus/greek_legal_ner")
    
    # Counter για τα tags στο train set μόνο
    train_tags_counter = Counter()
    unique_tags = set()  # Για να βρούμε τις μοναδικές τιμές
    
    print(f"\nΜέτρηση tags στο train set...")
    
    total_tokens = 0
    total_docs = 0
    
    for example in ds['test']:
        total_docs += 1
        ner_tags = example['ner']
        total_tokens += len(ner_tags)
        
        # Μετράμε κάθε tag και προσθέτουμε στο set
        for tag in ner_tags:
            train_tags_counter[tag] += 1
            unique_tags.add(tag)
    
    print(f"   📄 Έγγραφα: {total_docs:,}")
    print(f"   🔤 Tokens: {total_tokens:,}")
    print(f"   🏷️  Μοναδικά tags: {len(unique_tags)}")
    
    print("\n" + "="*70)
    print("📊 ΜΟΝΑΔΙΚΕΣ ΤΙΜΕΣ NER TAGS ΣΤΟ TRAIN SET")
    print("="*70)
    
    # Εκτύπωση όλων των μοναδικών tags
    print("🏷️  ΟΛΕΣ ΟΙ ΜΟΝΑΔΙΚΕΣ ΤΙΜΕΣ:")
    print("-" * 30)
    sorted_unique_tags = sorted(unique_tags)
    for i, tag in enumerate(sorted_unique_tags, 1):
        print(f"   {i:2d}. {tag}")
    
    print(f"\nΣυνολικές μοναδικές τιμές: {len(unique_tags)}")
    
    # Ταξινόμηση tags: πρώτα O, μετά B-, μετά I-
    sorted_tags = []
    
    # Πρώτα το O tag αν υπάρχει
    if 'O' in train_tags_counter:
        sorted_tags.append('O')
    
    # Μετά όλα τα B- tags
    b_tags = [tag for tag in train_tags_counter.keys() if tag.startswith('B-')]
    sorted_tags.extend(sorted(b_tags))
    
    # Τέλος όλα τα I- tags
    i_tags = [tag for tag in train_tags_counter.keys() if tag.startswith('I-')]
    sorted_tags.extend(sorted(i_tags))
    
    print("\n" + "="*70)
    print("📊 ΛΕΠΤΟΜΕΡΗ ΣΤΑΤΙΣΤΙΚΑ ΑΝΑ TAG")
    print("="*70)
    
    # Εκτύπωση αποτελεσμάτων
    print(f"{'TAG':<20} {'COUNT':<12} {'PERCENTAGE':<12}")
    print("-" * 45)
    
    total_all_tags = sum(train_tags_counter.values())
    
    for tag in sorted_tags:
        count = train_tags_counter[tag]
        percentage = (count / total_all_tags) * 100
        print(f"{tag:<20} {count:<12,} {percentage:<12.2f}%")
    
    print("-" * 45)
    print(f"{'TOTAL':<20} {total_all_tags:<12,} {'100.00%':<12}")
    
    # Ανάλυση ανά τύπο tag
    print(f"\n🏷️ ΑΝΑΛΥΣΗ ΑΝΑ ΤΥΠΟ TAG:")
    print("-" * 40)
    
    o_count = train_tags_counter.get('O', 0)
    b_count = sum(count for tag, count in train_tags_counter.items() if tag.startswith('B-'))
    i_count = sum(count for tag, count in train_tags_counter.items() if tag.startswith('I-'))
    
    print(f"O tags (Non-entity): {o_count:,} ({(o_count/total_all_tags)*100:.2f}%)")
    print(f"B- tags (Begin): {b_count:,} ({(b_count/total_all_tags)*100:.2f}%)")
    print(f"I- tags (Inside): {i_count:,} ({(i_count/total_all_tags)*100:.2f}%)")
    
    # Ανάλυση ανά entity type
    print(f"\n🏷️  ΑΝΑΛΥΣΗ ΑΝΑ ENTITY TYPE:")
    print("-" * 50)
    
    entity_types = set()
    for tag in train_tags_counter.keys():
        if tag.startswith('B-') or tag.startswith('I-'):
            entity_type = tag[2:]  # Αφαίρεση B- ή I-
            entity_types.add(entity_type)
    
    print(f"{'ENTITY TYPE':<15} {'B- TAGS':<10} {'I- TAGS':<10} {'TOTAL':<10}")
    print("-" * 50)
    
    for entity_type in sorted(entity_types):
        b_tag = f"B-{entity_type}"
        i_tag = f"I-{entity_type}"
        
        b_count = train_tags_counter.get(b_tag, 0)
        i_count = train_tags_counter.get(i_tag, 0)
        total = b_count + i_count
        
        print(f"{entity_type:<15} {b_count:<10,} {i_count:<10,} {total:<10,}")
    
    return dict(train_tags_counter)

if __name__ == "__main__":
    results = count_all_ner_tags()
    print(f"\n✅ Ολοκληρώθηκε η μέτρηση {sum(results.values()):,} συνολικών tags!")
