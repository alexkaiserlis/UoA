#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Γρήγορη δημιουργία περίληψης - Easy Summary Creator
"""

from eurovoc_summary_generator import EurovocSummaryGenerator

def create_summary_now():
    """Δημιουργεί περίληψη για συγκεκριμένο concept ID."""
    
    print("🚀 ΓΡΗΓΟΡΗ ΔΗΜΙΟΥΡΓΙΑ ΠΕΡΙΛΗΨΗΣ")
    print("="*50)
    
    # Δημιουργία generator
    generator = EurovocSummaryGenerator()
    
    # Φόρτωση δεδομένων
    print("📥 Φόρτωση δεδομένων...")
    generator.load_data()
    
    # Concept ID που θέλουμε
    concept_id = "40"  # Αλλαξε αυτό στο ID που θέλεις
    
    print(f"\n🎯 Δημιουργία περίληψης για concept: {concept_id}")
    
    try:
        # Έλεγχος αν υπάρχει το concept
        if concept_id not in generator.enhanced_mapping:
            print(f"❌ Concept {concept_id} δεν βρέθηκε!")
            return
        
        concept_data = generator.enhanced_mapping[concept_id]
        print(f"Concept: \"{concept_data['title']}\"")
        print(f"Category: {concept_data['category_full']}")
        
        # Δημιουργία περίληψης
        print(f"\n⏳ Δημιουργία περίληψης με OpenAI...")
        result = generator.create_summary_for_concept(
            concept_id=concept_id,
            summary_length='medium',  # short, medium, long
            model='gpt-4o-mini'       # φθηνότερο μοντέλο
        )
        
        # Αποθήκευση
        generator.save_summary(result)
        
        print(f"\n✅ ΕΠΙΤΥΧΙΑ!")
        print(f"📝 Word count: {result['metadata']['actual_word_count']}")
        print(f"🧠 Model: {result['metadata']['model_used']}")
        print(f"💾 Αποθηκεύτηκε: {generator.output_dir}/concept_{concept_id}_summary.json")
        
        print(f"\n📄 ΠΕΡΙΛΗΨΗ:")
        print("-" * 50)
        print(result['summary'])
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ ΣΦΑΛΜΑ: {str(e)}")

if __name__ == "__main__":
    create_summary_now()
