#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eurovision Concept Analysis Tool
===============================
Αναλύει όλες τις λέξεις που συνδέονται με ένα συγκεκριμένο Eurovision concept ID.
"""

import json
import os
from collections import Counter
from datetime import datetime

def analyze_concept_words(concept_id, save_to_file=True):
    """
    Αναλύει όλες τις λέξεις που περιέχουν το συγκεκριμένο Eurovision concept.
    
    Args:
        concept_id (str): Το Eurovision concept ID προς ανάλυση
        save_to_file (bool): Αν True, αποθηκεύει τα αποτελέσματα σε αρχείο
    """
    
    print(f"🔍 ΑΝΑΛΥΣΗ EUROVISION CONCEPT: {concept_id}")
    print("="*60)
    
    # Φόρτωση Eurovision mappings για τον τίτλο
    eurovoc_path = 'data/eurovoc_id_title_mappings.json'
    if not os.path.exists(eurovoc_path):
        print(f"❌ Δεν βρέθηκε το αρχείο: {eurovoc_path}")
        return
    
    with open(eurovoc_path, 'r', encoding='utf-8') as f:
        eurovoc_mappings = json.load(f)
    
    concept_title = eurovoc_mappings.get(concept_id, "UNKNOWN")
    print(f"📋 Concept Title: \"{concept_title}\"")
    print(f"🆔 Concept ID: {concept_id}")
    print()
    
    # Φόρτωση κύριου vocabulary
    vocab_path = 'data/eurlex_legal_vocabulary.json'
    if not os.path.exists(vocab_path):
        print(f"❌ Δεν βρέθηκε το αρχείο: {vocab_path}")
        return
    
    print("⏳ Φόρτωση vocabulary... (μπορεί να πάρει λίγο)")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)
    
    print(f"✅ Φορτώθηκαν {len(vocabulary):,} λέξεις")
    print()
    
    # Εύρεση λέξεων που περιέχουν το concept
    matching_words = []
    
    print("🔍 Αναζήτηση λέξεων με το concept...")
    for word, concepts in vocabulary.items():
        # Έλεγχος αν το concept_id υπάρχει στα concepts αυτής της λέξης
        for concept in concepts:
            if concept.get('id') == concept_id:
                matching_words.append(word)
                break
    
    # Στατιστικά
    total_words = len(matching_words)
    word_lengths = [len(word) for word in matching_words]
    
    results = {
        'analysis_info': {
            'concept_id': concept_id,
            'concept_title': concept_title,
            'analysis_date': datetime.now().isoformat(),
            'total_vocabulary_size': len(vocabulary)
        },
        'statistics': {
            'total_matching_words': total_words,
            'percentage_of_vocabulary': (total_words / len(vocabulary)) * 100 if vocabulary else 0,
            'word_length_stats': {
                'min_length': min(word_lengths) if word_lengths else 0,
                'max_length': max(word_lengths) if word_lengths else 0,
                'avg_length': sum(word_lengths) / len(word_lengths) if word_lengths else 0
            }
        },
        'word_analysis': {
            'all_words': sorted(matching_words),
            'word_frequency_by_length': dict(Counter(word_lengths)) if word_lengths else {},
            'sample_words': {
                'shortest': [w for w in matching_words if len(w) == min(word_lengths)] if word_lengths else [],
                'longest': [w for w in matching_words if len(w) == max(word_lengths)] if word_lengths else [],
                'most_common_length': sorted(matching_words, key=len)[len(matching_words)//2:len(matching_words)//2+5] if matching_words else []
            }
        }
    }
    
    # Εμφάνιση αποτελεσμάτων
    print("📊 ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΑΛΥΣΗΣ:")
    print("-" * 40)
    print(f"🎯 Concept: {concept_id} - \"{concept_title}\"")
    print(f"📝 Συνολικές λέξεις: {total_words:,}")
    print(f"📈 Ποσοστό vocabulary: {results['statistics']['percentage_of_vocabulary']:.2f}%")
    
    if word_lengths:
        print(f"📏 Μήκος λέξεων:")
        print(f"   • Ελάχιστο: {results['statistics']['word_length_stats']['min_length']} χαρακτήρες")
        print(f"   • Μέγιστο: {results['statistics']['word_length_stats']['max_length']} χαρακτήρες") 
        print(f"   • Μέσος όρος: {results['statistics']['word_length_stats']['avg_length']:.1f} χαρακτήρες")
    
    print()
    print("📋 ΔΕΙΓΜΑ ΛΕΞΕΩΝ:")
    print("-" * 40)
    
    # Εμφάνιση δείγματος λέξεων
    if matching_words:
        sample_size = min(20, len(matching_words))
        sample_words = sorted(matching_words)[:sample_size]
        
        for i, word in enumerate(sample_words, 1):
            print(f"{i:2d}. {word}")
        
        if len(matching_words) > sample_size:
            print(f"... και {len(matching_words) - sample_size:,} ακόμη λέξεις")
    else:
        print("❌ Δεν βρέθηκαν λέξεις για αυτό το concept!")
    
    # Αποθήκευση σε αρχείο
    if save_to_file and matching_words:
        output_filename = f"concept_{concept_id}_analysis.json"
        output_path = os.path.join('data', output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print()
        print(f"💾 Αποτελέσματα αποθηκεύτηκαν στο: {output_path}")
        
        # Δημιουργία και readable summary
        summary_filename = f"concept_{concept_id}_summary.txt"
        summary_path = os.path.join('data', summary_filename)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"EUROVISION CONCEPT ANALYSIS\n")
            f.write(f"{'='*50}\n")
            f.write(f"Concept ID: {concept_id}\n")
            f.write(f"Concept Title: {concept_title}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write(f"\nSTATISTICS:\n")
            f.write(f"{'='*30}\n")
            f.write(f"Total matching words: {total_words:,}\n")
            f.write(f"Percentage of vocabulary: {results['statistics']['percentage_of_vocabulary']:.2f}%\n")
            if word_lengths:
                f.write(f"Min word length: {results['statistics']['word_length_stats']['min_length']} chars\n")
                f.write(f"Max word length: {results['statistics']['word_length_stats']['max_length']} chars\n")
                f.write(f"Avg word length: {results['statistics']['word_length_stats']['avg_length']:.1f} chars\n")
            
            f.write(f"\nALL MATCHING WORDS:\n")
            f.write(f"{'='*30}\n")
            for i, word in enumerate(sorted(matching_words), 1):
                f.write(f"{i:4d}. {word}\n")
        
        print(f"📄 Readable summary αποθηκεύτηκε στο: {summary_path}")
    
    return results

def main():
    """Κύρια συνάρτηση με μενού επιλογών."""
    
    print("🇪🇺 EUROVISION CONCEPT ANALYZER")
    print("="*50)
    print()
    
    # Προτεινόμενα concepts για δοκιμή
    suggested_concepts = [
        ("1309", "import"),
        ("889", "State aid"), 
        ("192", "health control"),
        ("1318", "Germany"),
        ("2771", "originating product")
    ]
    
    print("🎯 ΠΡΟΤΕΙΝΟΜΕΝΑ CONCEPTS ΓΙΑ ΔΟΚΙΜΗ:")
    print("-" * 40)
    for i, (concept_id, title) in enumerate(suggested_concepts, 1):
        print(f"{i}. {concept_id} - \"{title}\"")
    
    print()
    choice = input("Επιλέξτε concept (1-5) ή εισάγετε δικό σας ID: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= 5:
        concept_id = suggested_concepts[int(choice)-1][0]
    else:
        concept_id = choice
    
    print()
    analyze_concept_words(concept_id)

if __name__ == "__main__":
    main()
