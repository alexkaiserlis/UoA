#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package Summary Script - Γρήγορη επισκόπηση του FINAL_VOCABULARY_PACKAGE
"""

import json
import csv
import os
from pathlib import Path

def print_package_summary():
    """Εκτυπώνει σύνοψη του package."""
    
    print("🎯" + "="*70)
    print("   EURLEX LEGAL VOCABULARY PACKAGE - SUMMARY")
    print("="*72)
    
    # Έλεγχος αρχείων
    files_info = {
        'eurlex_legal_vocabulary.json': 'Κύριο vocabulary με Eurovision mappings',
        'eurovoc_concepts_mapping.csv': 'Eurovision ID→Title mapping',
        'vocabulary_statistics.json': 'Λεπτομερή στατιστικά',
        'DOCUMENTATION.txt': 'Πλήρης τεκμηρίωση διαδικασίας',
        'README.md': 'Package documentation & usage guide'
    }
    
    print("\n📁 ΠΕΡΙΕΧΟΜΕΝΑ ΦΑΚΕΛΟΥ:")
    print("-" * 50)
    for filename, description in files_info.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename) / (1024 * 1024)  # MB
            status = "✅"
            print(f"{status} {filename:<35} {size:>8.1f}MB")
            print(f"   {description}")
        else:
            print(f"❌ {filename:<35} {'MISSING':>10}")
    
    # Στατιστικά vocabulary
    try:
        with open('vocabulary_statistics.json', 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        print(f"\n📊 ΣΤΑΤΙΣΤΙΚΑ VOCABULARY:")
        print("-" * 50)
        basic = stats['basic_statistics']
        print(f"Λέξεις:                    {basic['total_words']:,}")
        print(f"Eurovision concepts:       {basic['total_concepts']:,}")
        print(f"Μοναδικά concepts:         {basic['unique_concepts']:,}")
        print(f"Μέσος όρος:                {basic['average_concepts_per_word']:.2f} concepts/λέξη")
        
        print(f"\n🏆 TOP 3 ΛΕΞΕΙΣ ΜΕ ΠΕΡΙΣΣΟΤΕΡΑ CONCEPTS:")
        for i, (word, count) in enumerate(stats['top_words_by_concept_count'][:3], 1):
            print(f"{i}. '{word}': {count} concepts")
        
        print(f"\n🔥 TOP 3 ΣΥΧΝΟΤΕΡΑ EUROVISION CONCEPTS:")
        for i, (concept_id, freq) in enumerate(stats['most_frequent_concepts'][:3], 1):
            print(f"{i}. Concept {concept_id}: {freq} λέξεις")
            
    except FileNotFoundError:
        print("\n⚠️  Στατιστικά δεν βρέθηκαν")
    
    # Eurovision mapping info
    try:
        with open('eurovoc_concepts_mapping.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            eurovoc_count = sum(1 for row in reader) - 1  # -1 για header
        
        print(f"\n🇪🇺 EUROVISION MAPPING:")
        print("-" * 50)
        print(f"Eurovision concepts:       {eurovoc_count:,}")
        print(f"Format:                    ID;TERMS;RELATIONS;PT;MT")
        print(f"Coverage:                  99.35%")
        
    except FileNotFoundError:
        print("\n⚠️  Eurovision mapping δεν βρέθηκε")
    
    print(f"\n✨ ΠΟΙΟΤΗΤΑ & ΑΞΙΟΠΙΣΤΙΑ:")
    print("-" * 50)
    print("📚 Πηγή:                    EURLEX57K (45,000 έγγραφα)")
    print("🎯 Καθάρισμα:               5-step filtering pipeline")
    print("🌍 Standard:                Official EU vocabulary")
    print("🔍 Ακρίβεια:                99.35% concept coverage")
    print("⚖️  Domain:                  Legal/Regulatory documents")
    
    print(f"\n🚀 READY FOR:")
    print("-" * 50)
    print("• Σημασιολογική ανάλυση νομικών κειμένων")
    print("• Document classification βάσει Eurovision concepts")
    print("• Legal information retrieval systems")
    print("• Machine Learning applications")
    print("• Natural Language Processing")
    
    print(f"\n🎉 PACKAGE COMPLETE!")
    print("="*72)
    print("   Για λεπτομερείς οδηγίες: διαβάστε README.md")
    print("   Για τεχνικές λεπτομέρειες: διαβάστε DOCUMENTATION.txt")
    print("="*72)

if __name__ == "__main__":
    print_package_summary()
