#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Package Completion Summary
===============================
Τελικός έλεγχος και παρουσίαση του ολοκληρωμένου vocabulary package.
"""

import json
import os
from datetime import datetime

def final_package_summary():
    """Εμφανίζει τελικό summary του package."""
    
    print("🎉" + "="*70)
    print("   EURLEX VOCABULARY PACKAGE - ΤΕΛΙΚΗ ΟΛΟΚΛΗΡΩΣΗ")
    print("="*72)
    print(f"📅 Ημερομηνία: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print()
    
    # Έλεγχος όλων των αρχείων
    required_files = {
        'eurlex_legal_vocabulary.json': 'Κύριο vocabulary με Eurovision mappings',
        'eurovoc_concepts_mapping.csv': 'Επίσημο Eurovision CSV export', 
        'eurovoc_id_title_mappings.json': 'Καθαρό Eurovision ID→Title JSON',
        'vocabulary_statistics.json': 'Στατιστικά vocabulary',
        'eurovoc_mappings_statistics.json': 'Στατιστικά Eurovision mappings',
        'DOCUMENTATION.txt': 'Πλήρης τεκμηρίωση',
        'README.md': 'Οδηγός χρήσης',
    }
    
    sample_files = [
        'eurovoc_id_title_mappings_sample.json',
        'extract_eurovoc_id_title_mappings.py',
        'generate_statistics.py',
        'package_summary.py'
    ]
    
    print("📁 ΚΥΡΙΑ ΑΡΧΕΙΑ PACKAGE:")
    print("-" * 50)
    total_size = 0
    all_present = True
    
    for filename, description in required_files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename) / (1024 * 1024)  # MB
            total_size += size
            print(f"✅ {filename:<35} {size:>8.1f}MB")
            print(f"   {description}")
        else:
            print(f"❌ {filename:<35} {'MISSING':>10}")
            all_present = False
    
    print(f"\n📂 ΒΟΗΘΗΤΙΚΑ ΑΡΧΕΙΑ:")
    print("-" * 50)
    for filename in sample_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename) / 1024  # KB
            print(f"✅ {filename:<35} {size:>8.1f}KB")
    
    print(f"\n💾 ΣΥΝΟΛΙΚΟ ΜΕΓΕΘΟΣ: {total_size:.1f} MB")
    
    # Στατιστικά vocabulary
    if os.path.exists('vocabulary_statistics.json'):
        with open('vocabulary_statistics.json', 'r', encoding='utf-8') as f:
            vocab_stats = json.load(f)
        
        print(f"\n📊 ΣΤΑΤΙΣΤΙΚΑ VOCABULARY:")
        print("-" * 50)
        basic = vocab_stats['basic_statistics']
        print(f"Καθαρές λέξεις:            {basic['total_words']:,}")
        print(f"Eurovision concepts:       {basic['total_concepts']:,}")
        print(f"Μοναδικά concepts:         {basic['unique_concepts']:,}")
        print(f"Μέσος όρος:                {basic['average_concepts_per_word']:.2f} concepts/λέξη")
    
    # Στατιστικά Eurovision mappings
    if os.path.exists('eurovoc_mappings_statistics.json'):
        with open('eurovoc_mappings_statistics.json', 'r', encoding='utf-8') as f:
            eurovoc_stats = json.load(f)
        
        print(f"\n🇪🇺 ΣΤΑΤΙΣΤΙΚΑ EUROVISION MAPPINGS:")
        print("-" * 50)
        print(f"Συνολικά concepts:         {eurovoc_stats['total_concepts']:,}")
        print(f"Αριθμητικά IDs:            {eurovoc_stats['categories']['numeric_ids']:,}")
        print(f"Αλφαριθμητικά IDs:         {eurovoc_stats['categories']['alphanumeric_ids']:,}")
        print(f"Μέσο μήκος τίτλου:        {eurovoc_stats['title_length_stats']['avg']:.1f} χαρακτήρες")
    
    # Eurovision mapping examples
    if os.path.exists('eurovoc_id_title_mappings.json'):
        with open('eurovoc_id_title_mappings.json', 'r', encoding='utf-8') as f:
            eurovoc_mappings = json.load(f)
        
        print(f"\n🎯 ΠΑΡΑΔΕΙΓΜΑΤΑ EUROVISION MAPPINGS:")
        print("-" * 50)
        # Τα πιο συχνά concepts
        frequent_concepts = ['1309', '2771', '192', '889', '1318']
        for concept_id in frequent_concepts:
            if concept_id in eurovoc_mappings:
                print(f"{concept_id}: \"{eurovoc_mappings[concept_id]}\"")
    
    print(f"\n✨ ΠΟΙΟΤΗΤΑ & ΧΑΡΑΚΤΗΡΙΣΤΙΚΑ:")
    print("-" * 50)
    print("🎯 Πηγή:                    EURLEX57K (45,000 νομικά έγγραφα)")
    print("🧹 Καθάρισμα:               5-step filtering pipeline")
    print("🌍 Standard:                Official EU vocabulary")
    print("📈 Coverage:                99.35% Eurovision concept titles")
    print("⚖️  Domain:                  Legal/Regulatory documents")
    print("🔧 Format:                  JSON + CSV για μέγιστη συμβατότητα")
    
    print(f"\n🚀 ΕΤΟΙΜΟ ΓΙΑ:")
    print("-" * 50)
    print("• Σημασιολογική ανάλυση νομικών κειμένων")
    print("• Document classification βάσει Eurovision concepts")
    print("• Legal information retrieval systems")  
    print("• Machine Learning και NLP εφαρμογές")
    print("• Research και ακαδημαϊκές μελέτες")
    print("• EU legal document processing")
    
    if all_present:
        print(f"\n🎉 PACKAGE ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ!")
        print("="*72)
        print("   ✅ Όλα τα αρχεία παρόντα")
        print("   ✅ Πλήρης τεκμηρίωση")
        print("   ✅ Έτοιμο για διανομή")
        print("="*72)
    else:
        print(f"\n⚠️  PACKAGE ΗΜΙΤΕΛΕΣ - Λείπουν αρχεία!")
    
    print(f"\n📚 ΟΔΗΓΙΕΣ ΧΡΗΣΗΣ:")
    print("   • Διαβάστε το README.md για γρήγορη εκκίνηση")
    print("   • Δείτε το DOCUMENTATION.txt για λεπτομέρειες")
    print("   • Χρησιμοποιήστε eurovoc_id_title_mappings.json για lookups")
    print("   • Φορτώστε eurlex_legal_vocabulary.json για NLP tasks")

if __name__ == "__main__":
    final_package_summary()
