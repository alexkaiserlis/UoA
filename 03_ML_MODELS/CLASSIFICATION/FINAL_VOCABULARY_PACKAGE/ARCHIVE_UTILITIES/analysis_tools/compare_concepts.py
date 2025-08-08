#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concept Comparison Tool
======================
Συγκρίνει λέξεις μεταξύ διαφορετικών Eurovision concepts.
"""

import json
import os
from datetime import datetime

def compare_concepts():
    """Συγκρίνει τα αποτελέσματα από διαφορετικά concepts."""
    
    print("📊 ΣΥΓΚΡΙΣΗ EUROVISION CONCEPTS")
    print("="*50)
    
    # Εύρεση υπαρχόντων analysis files
    data_dir = 'data'
    analysis_files = [f for f in os.listdir(data_dir) if f.startswith('concept_') and f.endswith('_analysis.json')]
    
    if not analysis_files:
        print("❌ Δεν βρέθηκαν analysis files. Εκτελέστε πρώτα το analyze_concept.py")
        return
    
    concepts_data = {}
    
    # Φόρτωση δεδομένων από όλα τα analysis files
    for filename in analysis_files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            concept_id = data['analysis_info']['concept_id']
            concepts_data[concept_id] = data
    
    print(f"📁 Βρέθηκαν {len(concepts_data)} analyzed concepts:")
    print("-" * 40)
    
    # Στατιστικά σύγκρισης
    comparison_table = []
    
    for concept_id, data in concepts_data.items():
        title = data['analysis_info']['concept_title']
        total_words = data['statistics']['total_matching_words']
        percentage = data['statistics']['percentage_of_vocabulary']
        avg_length = data['statistics']['word_length_stats']['avg_length']
        
        comparison_table.append({
            'id': concept_id,
            'title': title,
            'words': total_words,
            'percentage': percentage,
            'avg_length': avg_length
        })
        
        print(f"🎯 {concept_id}: \"{title}\" - {total_words:,} λέξεις ({percentage:.1f}%)")
    
    # Ταξινόμηση κατά αριθμό λέξεων
    comparison_table.sort(key=lambda x: x['words'], reverse=True)
    
    print(f"\n📊 ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ (ταξινομημένος κατά αριθμό λέξεων):")
    print("="*80)
    print(f"{'Rank':<4} | {'ID':<6} | {'Λέξεις':<8} | {'Ποσοστό':<8} | {'Μ.Ο. Μήκος':<10} | {'Τίτλος':<20}")
    print("-"*80)
    
    for i, concept in enumerate(comparison_table, 1):
        print(f"{i:<4} | {concept['id']:<6} | {concept['words']:<8,} | {concept['percentage']:<7.1f}% | {concept['avg_length']:<9.1f} | {concept['title']:<20}")
    
    # Εύρεση κοινών λέξεων
    if len(concepts_data) >= 2:
        print(f"\n🔗 ΑΝΑΛΥΣΗ ΚΟΙΝΩΝ ΛΕΞΕΩΝ:")
        print("-" * 40)
        
        concept_words = {}
        for concept_id, data in concepts_data.items():
            concept_words[concept_id] = set(data['word_analysis']['all_words'])
        
        concept_ids = list(concept_words.keys())
        
        # Κοινές λέξεις μεταξύ όλων
        common_words = concept_words[concept_ids[0]]
        for concept_id in concept_ids[1:]:
            common_words = common_words.intersection(concept_words[concept_id])
        
        print(f"📝 Κοινές λέξεις σε όλα τα concepts: {len(common_words):,}")
        if len(common_words) <= 20:
            print(f"   {sorted(list(common_words))}")
        else:
            sample_common = sorted(list(common_words))[:10]
            print(f"   Δείγμα: {sample_common} ...")
        
        # Ζεύγη concepts
        for i, concept1 in enumerate(concept_ids):
            for concept2 in concept_ids[i+1:]:
                intersection = concept_words[concept1].intersection(concept_words[concept2])
                title1 = concepts_data[concept1]['analysis_info']['concept_title']
                title2 = concepts_data[concept2]['analysis_info']['concept_title']
                
                print(f"🔗 {concept1} (\"{title1}\") ∩ {concept2} (\"{title2}\"): {len(intersection):,} κοινές λέξεις")
    
    # Δημιουργία comparison report
    report_data = {
        'comparison_info': {
            'analysis_date': datetime.now().isoformat(),
            'total_concepts_analyzed': len(concepts_data),
            'total_vocabulary_size': 40431  # από το vocabulary
        },
        'concepts_summary': comparison_table,
        'detailed_comparison': {
            'most_words': comparison_table[0] if comparison_table else None,
            'least_words': comparison_table[-1] if comparison_table else None,
            'average_words_per_concept': sum(c['words'] for c in comparison_table) / len(comparison_table) if comparison_table else 0,
            'average_coverage_percentage': sum(c['percentage'] for c in comparison_table) / len(comparison_table) if comparison_table else 0
        }
    }
    
    # Αποθήκευση σύγκρισης
    comparison_file = os.path.join(data_dir, 'concepts_comparison.json')
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Σύγκριση αποθηκεύτηκε στο: {comparison_file}")
    
    print(f"\n✨ ΣΥΜΠΕΡΑΣΜΑΤΑ:")
    print("-" * 40)
    if comparison_table:
        best = comparison_table[0]
        worst = comparison_table[-1]
        
        print(f"🥇 Πιο δημοφιλές concept: {best['id']} (\"{best['title']}\") με {best['words']:,} λέξεις")
        print(f"🥉 Λιγότερο δημοφιλές: {worst['id']} (\"{worst['title']}\") με {worst['words']:,} λέξεις")
        print(f"📊 Μέσος όρος λέξεων ανά concept: {report_data['detailed_comparison']['average_words_per_concept']:.0f}")
        print(f"📈 Μέση κάλυψη vocabulary: {report_data['detailed_comparison']['average_coverage_percentage']:.1f}%")

if __name__ == "__main__":
    compare_concepts()
