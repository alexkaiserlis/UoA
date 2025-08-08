#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script για δημιουργία στατιστικών summary του τελικού vocabulary
"""

import json

def create_statistics_summary():
    """Δημιουργεί λεπτομερή στατιστικά του τελικού vocabulary."""
    
    # Φόρτωση vocabulary
    with open('../eurlex_train_vocabulary_ENRICHED_NO_NUMBERS.json', 'r', encoding='utf-8') as file:
        vocab = json.load(file)
    
    # Βασικά στατιστικά
    total_words = len(vocab)
    total_concepts = sum(len(concepts) for concepts in vocab.values())
    avg_concepts = total_concepts / total_words
    
    # Μοναδικά concepts
    unique_concepts = set()
    for concepts in vocab.values():
        for concept in concepts:
            unique_concepts.add(concept['id'])
    
    # Top λέξεις με περισσότερα concepts
    words_by_concept_count = sorted(
        [(word, len(concepts)) for word, concepts in vocab.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Frequency των concepts
    concept_frequency = {}
    for concepts in vocab.values():
        for concept in concepts:
            concept_id = concept['id']
            concept_frequency[concept_id] = concept_frequency.get(concept_id, 0) + 1
    
    most_frequent_concepts = sorted(
        concept_frequency.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Δημιουργία summary
    summary = {
        'basic_statistics': {
            'total_words': total_words,
            'total_concepts': total_concepts,
            'average_concepts_per_word': avg_concepts,
            'unique_concepts': len(unique_concepts)
        },
        'top_words_by_concept_count': words_by_concept_count,
        'most_frequent_concepts': most_frequent_concepts,
        'sample_vocabulary_entries': {
            word: vocab[word][:3] for word in list(vocab.keys())[:5]
        }
    }
    
    # Αποθήκευση
    with open('vocabulary_statistics.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    
    # Εκτύπωση
    print("ΣΤΑΤΙΣΤΙΚΑ ΤΕΛΙΚΟΥ VOCABULARY")
    print("="*40)
    print(f"Λέξεις: {total_words:,}")
    print(f"Συνολικά concepts: {total_concepts:,}")
    print(f"Μέσος όρος: {avg_concepts:.2f} concepts/λέξη")
    print(f"Μοναδικά concepts: {len(unique_concepts):,}")
    print(f"\nTop 5 λέξεις με περισσότερα concepts:")
    for word, count in words_by_concept_count[:5]:
        print(f"  {word}: {count}")
    print(f"\nTop 5 συχνότερα concepts:")
    for concept_id, freq in most_frequent_concepts[:5]:
        # Βρίσκουμε το πρώτο title για αυτό το concept
        title = "Unknown"
        for concepts in vocab.values():
            for c in concepts:
                if c['id'] == concept_id:
                    title = c['title']
                    break
            if title != "Unknown":
                break
        print(f"  {concept_id} ({title}): {freq} λέξεις")

if __name__ == "__main__":
    create_statistics_summary()
