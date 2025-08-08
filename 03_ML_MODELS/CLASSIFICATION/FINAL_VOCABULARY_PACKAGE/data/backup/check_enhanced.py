#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Έλεγχος του enhanced concepts αρχείου
"""

import json
import os

def check_enhanced_concepts():
    """Έλεγχος του επεκτάσμένου αρχείου"""
    data_dir = os.path.dirname(os.path.abspath(__file__))
    enhanced_path = os.path.join(data_dir, "eurovoc_enhanced_concepts_with_eurlex.json")
    
    try:
        with open(enhanced_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 Συνολικά concepts: {len(data)}")
        
        # Μετρήματα
        concepts_with_alt_labels = 0
        concepts_with_parents = 0
        concepts_with_eurlex_source = 0
        new_eurlex_concepts = 0
        total_alt_labels = 0
        
        sample_enhanced_concepts = []
        
        for concept_id, concept_data in data.items():
            if concept_data.get('alt_labels'):
                concepts_with_alt_labels += 1
                total_alt_labels += len(concept_data['alt_labels'])
                
                if len(sample_enhanced_concepts) < 3:
                    sample_enhanced_concepts.append((concept_id, concept_data))
                    
            if concept_data.get('parents'):
                concepts_with_parents += 1
                
            if concept_data.get('eurlex_source'):
                concepts_with_eurlex_source += 1
                
            if concept_data.get('source') == 'EURLEX57K':
                new_eurlex_concepts += 1
        
        print(f"🏷️  Concepts με alt_labels: {concepts_with_alt_labels}")
        print(f"🌳 Concepts με parents: {concepts_with_parents}")
        print(f"📚 Concepts βελτιωμένα από EURLEX57K: {concepts_with_eurlex_source}")
        print(f"🆕 Νέα concepts από EURLEX57K: {new_eurlex_concepts}")
        print(f"🔤 Συνολικά alt_labels: {total_alt_labels}")
        
        print("\n📋 Δείγματα enhanced concepts:")
        for i, (concept_id, concept_data) in enumerate(sample_enhanced_concepts):
            print(f"\n{i+1}. ID: {concept_id}")
            print(f"   Title: {concept_data.get('title', 'N/A')}")
            print(f"   Alt labels: {concept_data.get('alt_labels', [])[:3]}...")
            print(f"   Parents: {concept_data.get('parents', [])[:3]}...")
            
    except Exception as e:
        print(f"❌ Σφάλμα: {e}")

if __name__ == "__main__":
    check_enhanced_concepts()
