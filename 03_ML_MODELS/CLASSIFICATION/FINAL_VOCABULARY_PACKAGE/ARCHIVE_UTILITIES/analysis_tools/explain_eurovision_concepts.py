#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eurovision Concepts Explanation
===============================
Εξήγηση των πιο συχνών Eurovision concepts από το vocabulary.
"""

import json
import os

def explain_eurovision_concepts():
    """Εξηγεί τι είναι τα Eurovision concepts και γιατί είναι αριθμοί."""
    
    print("🇪🇺" + "="*70)
    print("   EUROVISION CONCEPTS - ΕΞΗΓΗΣΗ")
    print("="*72)
    print()
    
    print("❓ ΓΙΑΤΙ ΕΙΝΑΙ ΑΡΙΘΜΟΙ ΤΑ EUROVISION CONCEPTS;")
    print("-" * 60)
    print("Τα Eurovision concepts είναι αριθμοί γιατί είναι τα επίσημα")
    print("classification IDs από το Eurovision thesaurus της Ευρωπαϊκής Ένωσης!")
    print()
    print("🎯 Κάθε αριθμός αντιστοιχεί σε ένα συγκεκριμένο νομικό/πολιτικό θέμα")
    print("🎯 Χρησιμοποιούνται για την κατηγοριοποίηση όλων των EU documents")
    print("🎯 Είναι διεθνής standard για EU legal information systems")
    print()
    
    # Φόρτωση Eurovision mappings
    eurovoc_path = os.path.join('..', 'data', 'eurovoc_id_title_mappings.json')
    if not os.path.exists(eurovoc_path):
        eurovoc_path = os.path.join('data', 'eurovoc_id_title_mappings.json')
    
    if os.path.exists(eurovoc_path):
        with open(eurovoc_path, 'r', encoding='utf-8') as f:
            eurovoc_mappings = json.load(f)
        
        # Τα πιο συχνά concepts από τα statistics
        most_frequent = [
            ('1309', 10233, 'import'),
            ('2771', 9532, 'originating product'), 
            ('192', 7953, 'health control'),
            ('889', 7906, 'State aid'),
            ('1318', 7384, 'Germany'),
            ('1085', 7339, 'third country'),
            ('5451', 7263, 'plant health control'),
            ('1519', 7102, 'directive'),
            ('2300', 6945, 'approximation of laws'),
            ('863', 6847, 'free movement of goods')
        ]
        
        print("🎯 ΤΑ 10 ΠΙΟ ΣΥΧΝΑ EUROVISION CONCEPTS ΣΤΟ VOCABULARY:")
        print("="*70)
        print(f"{'ID':<6} | {'Συχνότητα':<10} | {'Τίτλος':<30}")
        print("-"*70)
        
        for concept_id, frequency, expected_title in most_frequent:
            actual_title = eurovoc_mappings.get(concept_id, expected_title)
            print(f"{concept_id:<6} | {frequency:>10,} | {actual_title:<30}")
    
    print()
    print("📋 ΠΑΡΑΔΕΙΓΜΑΤΑ ΚΑΤΗΓΟΡΙΩΝ:")
    print("-" * 60)
    print("• 1000-1999: Οικονομικά θέματα (financing, trade, etc.)")
    print("• 2000-2999: Εμπόριο και βιομηχανία")  
    print("• 100-999:   Κοινωνικές πολιτικές")
    print("• 5000+:     Περιβάλλον και υγεία")
    print("• 800-899:   Εσωτερική αγορά")
    
    print()
    print("💡 ΓΙΑΤΙ ΧΡΗΣΙΜΟΠΟΙΟΥΝΤΑΙ ΑΡΙΘΜΟΙ;")
    print("-" * 60)
    print("✅ Διεθνής συμβατότητα (ανεξάρτητα από γλώσσα)")
    print("✅ Σταθερότητα στο χρόνο (οι τίτλοι μπορεί να αλλάξουν)")
    print("✅ Συστηματική οργάνωση (hierarchical structure)")
    print("✅ Αποδοτικότητα σε databases και systems")
    print("✅ Ακρίβεια στην κατηγοριοποίηση")
    
    print()
    print("🔗 ΠΛΗΡΟΦΟΡΙΕΣ:")
    print("-" * 60)
    print("• Eurovision = EUROpean VOCabulary")
    print("• Διαχειρίζεται από το Publications Office της EU")
    print("• Χρησιμοποιείται σε EUR-Lex, CELLAR, και άλλα EU systems")
    print("• Διαθέσιμο σε 24+ γλώσσες της EU")
    
    print()
    print("📊 ΣΤΟ ΔΙΚΟ ΜΑΣ VOCABULARY:")
    print("-" * 60)
    print("• 4,108 μοναδικά Eurovision concepts")
    print("• 3,233,099 συνολικά mappings")
    print("• 99.35% coverage των concept titles")
    print("• Όλα προέρχονται από πραγματικά EU νομικά έγγραφα")

if __name__ == "__main__":
    explain_eurovision_concepts()
