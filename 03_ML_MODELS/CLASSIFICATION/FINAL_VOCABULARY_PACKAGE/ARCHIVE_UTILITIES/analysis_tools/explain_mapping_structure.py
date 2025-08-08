#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eurovision Enhanced Mapping Structure Analyzer
==============================================
Εξηγεί αναλυτικά τη δομή του εμπλουτισμένου Eurovision mapping.
"""

import json

def explain_enhanced_mapping_structure():
    """Εξηγεί αναλυτικά κάθε πεδίο του εμπλουτισμένου mapping."""
    
    print("📚 ΑΝΑΛΥΣΗ ΔΟΜΗΣ ΕΜΠΛΟΥΤΙΣΜΕΝΟΥ EUROVISION MAPPING")
    print("="*65)
    
    # Φόρτωση του mapping
    with open('data/eurovoc_enhanced_mapping.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("🏗️  ΣΥΝΟΛΙΚΗ ΔΟΜΗ ΑΡΧΕΙΟΥ:")
    print("-" * 50)
    print("Το JSON έχει 2 κύρια τμήματα:")
    print("  1. 📊 metadata - Μεταδεδομένα και στατιστικά")
    print("  2. 🎯 concepts - Τα πραγματικά Eurovision concepts")
    print()
    
    # Ανάλυση metadata
    metadata = data['metadata']
    print("📊 ΤΜΗΜΑ METADATA:")
    print("-" * 40)
    print("Περιέχει 3 υποτμήματα:")
    print()
    
    print("  📅 creation_info:")
    creation_info = metadata['creation_info']
    for key, value in creation_info.items():
        print(f"     • {key}: {value}")
    print()
    
    print("  📈 statistics:")
    stats = metadata['statistics']
    print(f"     • direct_terms: {stats['direct_terms']} (κύριοι όροι)")
    print(f"     • redirect_terms: {stats['redirect_terms']} (redirects/συνώνυμα)")
    print(f"     • top_categories: Top {len(stats['top_categories'])} κατηγορίες")
    print()
    
    print("  📋 structure_info:")
    struct_info = metadata['structure_info']
    print(f"     • description: {struct_info['description']}")
    print("     • fields: Εξήγηση όλων των πεδίων των concepts")
    print()
    
    # Ανάλυση concepts
    concepts = data['concepts']
    print("🎯 ΤΜΗΜΑ CONCEPTS:")
    print("-" * 40)
    print(f"Περιέχει {len(concepts):,} Eurovision concepts")
    print("Κάθε concept έχει τη δομή:")
    print()
    
    # Παράδειγμα concept
    sample_id = list(concepts.keys())[0]
    sample_concept = concepts[sample_id]
    
    print(f"📍 ΠΑΡΑΔΕΙΓΜΑ CONCEPT: {sample_id}")
    print("-" * 50)
    
    fields_explanation = {
        'title': {
            'description': 'Το όνομα/τίτλος του concept',
            'example': sample_concept['title'],
            'type': 'string',
            'purpose': 'Η κύρια ονομασία που χρησιμοποιείται'
        },
        'category_code': {
            'description': 'Αριθμητικός κωδικός θεματικής κατηγορίας',
            'example': sample_concept['category_code'],
            'type': 'string (αριθμός)',
            'purpose': 'Ομαδοποίηση κατά θεματική περιοχή'
        },
        'category_name': {
            'description': 'Όνομα θεματικής κατηγορίας',
            'example': sample_concept['category_name'],
            'type': 'string',
            'purpose': 'Ανθρώπινα κατανοητή περιγραφή κατηγορίας'
        },
        'category_full': {
            'description': 'Πλήρης κατηγορία (κωδικός + όνομα)',
            'example': sample_concept['category_full'],
            'type': 'string',
            'purpose': 'Ολοκληρωμένη αναφορά στην κατηγορία'
        },
        'is_redirect': {
            'description': 'Αν το concept είναι redirect σε άλλο',
            'example': sample_concept['is_redirect'],
            'type': 'boolean (true/false)',
            'purpose': 'Διαχείριση συνωνύμων και εναλλακτικών όρων'
        },
        'preferred_term': {
            'description': 'Κύριος όρος (αν υπάρχει redirect)',
            'example': sample_concept['preferred_term'],
            'type': 'string ή null',
            'purpose': 'Ο "σωστός" όρος προς χρήση'
        },
        'relation_type': {
            'description': 'Τύπος σχέσης (συνήθως "USE")',
            'example': sample_concept['relation_type'],
            'type': 'string ή null',
            'purpose': 'Τεχνική πληροφορία για τη σχέση'
        }
    }
    
    for field, info in fields_explanation.items():
        print(f"🔸 {field}:")
        print(f"   📝 Τι είναι: {info['description']}")
        print(f"   📊 Τύπος: {info['type']}")
        print(f"   🎯 Σκοπός: {info['purpose']}")
        print(f"   💡 Παράδειγμα: {info['example']}")
        print()
    
    # Παραδείγματα διαφορετικών τύπων concepts
    print("🔍 ΠΑΡΑΔΕΙΓΜΑΤΑ ΔΙΑΦΟΡΕΤΙΚΩΝ ΤΥΠΩΝ:")
    print("-" * 50)
    
    # Εύρεση παραδειγμάτων
    direct_example = None
    redirect_example = None
    
    for concept_id, concept_data in concepts.items():
        if not redirect_example and concept_data['is_redirect']:
            redirect_example = (concept_id, concept_data)
        if not direct_example and not concept_data['is_redirect']:
            direct_example = (concept_id, concept_data)
        if direct_example and redirect_example:
            break
    
    print("1️⃣ ΚΥΡΙΟΣ ΟΡΟΣ (direct term):")
    if direct_example:
        concept_id, concept_data = direct_example
        print(f"   ID: {concept_id}")
        print(f"   Title: \"{concept_data['title']}\"")
        print(f"   Category: {concept_data['category_full']}")
        print(f"   Is Redirect: {concept_data['is_redirect']}")
        print("   ➜ Αυτό είναι ένας κύριος όρος, χρησιμοποιείται άμεσα")
    print()
    
    print("2️⃣ REDIRECT/ΣΥΝΩΝΥΜΟ:")
    if redirect_example:
        concept_id, concept_data = redirect_example
        print(f"   ID: {concept_id}")
        print(f"   Title: \"{concept_data['title']}\"")
        print(f"   Category: {concept_data['category_full']}")
        print(f"   Is Redirect: {concept_data['is_redirect']}")
        print(f"   Preferred Term: \"{concept_data['preferred_term']}\"")
        print(f"   ➜ Όταν κάποιος αναζητά \"{concept_data['title']}\"")
        print(f"     θα πρέπει να χρησιμοποιήσει \"{concept_data['preferred_term']}\"")
    print()
    
    # Στατιστικά κατηγοριών
    categories_stats = {}
    redirect_stats = {'direct': 0, 'redirects': 0}
    
    for concept_data in concepts.values():
        category = concept_data['category_full']
        categories_stats[category] = categories_stats.get(category, 0) + 1
        
        if concept_data['is_redirect']:
            redirect_stats['redirects'] += 1
        else:
            redirect_stats['direct'] += 1
    
    print("📊 ΣΤΑΤΙΣΤΙΚΑ ΧΡΗΣΗΣ:")
    print("-" * 40)
    print(f"Συνολικά concepts: {len(concepts):,}")
    print(f"Κύριοι όροι: {redirect_stats['direct']:,} ({redirect_stats['direct']/len(concepts)*100:.1f}%)")
    print(f"Redirects: {redirect_stats['redirects']:,} ({redirect_stats['redirects']/len(concepts)*100:.1f}%)")
    print(f"Διαφορετικές κατηγορίες: {len(categories_stats):,}")
    print()
    
    print("🏆 ΤΟΠ 5 ΚΑΤΗΓΟΡΙΕΣ:")
    print("-" * 40)
    top_categories = sorted(categories_stats.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (category, count) in enumerate(top_categories, 1):
        print(f"{i}. {category}: {count:,} concepts")
    print()
    
    print("💡 ΠΡΑΚΤΙΚΗ ΧΡΗΣΗ:")
    print("-" * 40)
    print("✅ Για αναζήτηση concept:")
    print("   1. Ψάξτε το ID στο mapping")
    print("   2. Αν is_redirect = true, χρησιμοποιήστε το preferred_term")
    print("   3. Αν is_redirect = false, χρησιμοποιήστε το title")
    print()
    print("✅ Για θεματική ομαδοποίηση:")
    print("   1. Χρησιμοποιήστε το category_code για ομαδοποίηση")
    print("   2. Χρησιμοποιήστε το category_name για labels")
    print()
    print("✅ Για περιλήψεις:")
    print("   1. Ομαδοποιήστε concepts κατά category_code")
    print("   2. Δημιουργήστε θεματικές ενότητες")
    print("   3. Χρησιμοποιήστε τα preferred_terms για κύριους όρους")

if __name__ == "__main__":
    explain_enhanced_mapping_structure()
