#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script για επέκταση του concepts αρχείου με πληροφορίες από το EURLEX57K
"""

import json
import os
from typing import Dict, Any

def load_eurlex57k_data(file_path: str) -> Dict[str, Any]:
    """Φόρτωση δεδομένων από το EURLEX57K αρχείο"""
    print("🔄 Φόρτωση EURLEX57K δεδομένων...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Φορτώθηκαν {len(data)} concepts από EURLEX57K")
        return data
    except Exception as e:
        print(f"❌ Σφάλμα κατά τη φόρτωση EURLEX57K: {e}")
        return {}

def load_existing_concepts(file_path: str) -> Dict[str, Any]:
    """Φόρτωση υπάρχοντος concepts αρχείου"""
    print("🔄 Φόρτωση υπάρχοντος concepts αρχείου...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Φορτώθηκαν {len(data)} concepts από υπάρχον αρχείο")
        return data
    except Exception as e:
        print(f"❌ Σφάλμα κατά τη φόρτωση υπάρχοντος αρχείου: {e}")
        return {}

def enhance_concepts(eurlex_data: Dict[str, Any], existing_concepts: Dict[str, Any]) -> Dict[str, Any]:
    """Επέκταση των concepts με πληροφορίες από EURLEX57K"""
    print("🔄 Επέκταση concepts με EURLEX57K δεδομένα...")
    
    enhanced_concepts = existing_concepts.copy()
    new_concepts_added = 0
    existing_concepts_enhanced = 0
    
    for concept_id, concept_data in eurlex_data.items():
        # Αντιστοίχιση concept_id (από EURLEX57K χρησιμοποιεί το concept_id string)
        actual_id = concept_data.get('concept_id', concept_id)
        
        # Δημιουργία enhanced δεδομένων
        enhanced_data = {
            "title": concept_data.get('label', ''),
            "alt_labels": concept_data.get('alt_labels', []),
            "parents": concept_data.get('parents', []),
            "concept_id": actual_id,
            "category_code": None,  # Θα μπορούσε να συμπληρωθεί από mapping
            "category_name": None,
            "category_full": None,
            "is_redirect": False,
            "preferred_term": None,
            "relation_type": None,
            "source": "EURLEX57K"
        }
        
        # Αν το concept υπάρχει ήδη, το βελτιώνουμε
        if actual_id in enhanced_concepts:
            existing = enhanced_concepts[actual_id]
            
            # Συνδυασμός alt_labels
            existing_alt_labels = existing.get('alt_labels', [])
            new_alt_labels = concept_data.get('alt_labels', [])
            combined_alt_labels = list(set(existing_alt_labels + new_alt_labels))
            
            # Διατήρηση υπαρχόντων δεδομένων και προσθήκη νέων
            enhanced_concepts[actual_id].update({
                "alt_labels": combined_alt_labels,
                "parents": concept_data.get('parents', existing.get('parents', [])),
                "eurlex_label": concept_data.get('label', ''),
                "eurlex_source": True
            })
            existing_concepts_enhanced += 1
            
        else:
            # Προσθήκη νέου concept
            enhanced_concepts[actual_id] = enhanced_data
            new_concepts_added += 1
    
    print(f"✅ Προστέθηκαν {new_concepts_added} νέα concepts")
    print(f"✅ Βελτιώθηκαν {existing_concepts_enhanced} υπάρχοντα concepts")
    print(f"📊 Συνολικά concepts: {len(enhanced_concepts)}")
    
    return enhanced_concepts

def save_enhanced_concepts(enhanced_concepts: Dict[str, Any], output_path: str):
    """Αποθήκευση των enhanced concepts"""
    print(f"💾 Αποθήκευση enhanced concepts στο {output_path}...")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_concepts, f, ensure_ascii=False, indent=2)
        print("✅ Επιτυχής αποθήκευση!")
    except Exception as e:
        print(f"❌ Σφάλμα κατά την αποθήκευση: {e}")

def main():
    """Κύρια συνάρτηση"""
    print("🚀 ΕΠΕΚΤΑΣΗ CONCEPTS ΜΕ EURLEX57K ΔΕΔΟΜΕΝΑ")
    print("=" * 50)
    
    # Διαδρομές αρχείων
    data_dir = os.path.dirname(os.path.abspath(__file__))
    eurlex_path = os.path.join(data_dir, "EURLEX57K.json")
    existing_concepts_path = os.path.join(data_dir, "eurovoc_enhanced_concepts_only.json")
    output_path = os.path.join(data_dir, "eurovoc_enhanced_concepts_with_eurlex.json")
    
    # Φόρτωση δεδομένων
    eurlex_data = load_eurlex57k_data(eurlex_path)
    if not eurlex_data:
        print("❌ Δεν μπόρεσα να φορτώσω τα EURLEX57K δεδομένα")
        return
    
    existing_concepts = load_existing_concepts(existing_concepts_path)
    if not existing_concepts:
        print("⚠️ Δεν βρέθηκε υπάρχον concepts αρχείο, θα δημιουργήσω νέο")
        existing_concepts = {}
    
    # Επέκταση concepts
    enhanced_concepts = enhance_concepts(eurlex_data, existing_concepts)
    
    # Αποθήκευση
    save_enhanced_concepts(enhanced_concepts, output_path)
    
    print("\n🎉 ΟΛΟΚΛΗΡΩΣΗ ΕΠΕΚΤΑΣΗΣ!")
    print(f"📁 Αρχείο αποθηκεύτηκε: {output_path}")

if __name__ == "__main__":
    main()
