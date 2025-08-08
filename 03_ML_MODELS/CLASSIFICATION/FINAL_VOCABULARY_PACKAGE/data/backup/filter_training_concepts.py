#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script για δημιουργία καθαρής έκδοσης concepts που περιέχει μόνο 
όρους που αναφέρονται στο EURLEX training data
"""

import json
import os
from typing import Dict, Any

def load_concepts_with_paths(file_path: str) -> Dict[str, Any]:
    """Φόρτωση του concepts αρχείου με paths"""
    print("🔄 Φόρτωση concepts με paths...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Φορτώθηκαν {len(data)} concepts")
        return data
    except Exception as e:
        print(f"❌ Σφάλμα κατά τη φόρτωση: {e}")
        return {}

def load_train_labels_concepts(train_labels_path: str) -> set:
    """Φόρτωση όλων των concepts που αναφέρονται στο EURLEX training"""
    print("🔄 Φόρτωση EURLEX training labels για εξαγωγή concepts...")
    
    eurlex_concepts = set()
    
    try:
        with open(train_labels_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    label_data = json.loads(line.strip())
                    
                    # Εξαγωγή όλων των labels από όλα τα levels
                    for level in ['level_1_label', 'level_2_label', 'level_3_label', 'level_4_label', 'level_5_label']:
                        label = label_data.get(level, '').strip()
                        if label:
                            eurlex_concepts.add(label)
                            
                except json.JSONDecodeError:
                    continue
                    
                if line_num % 50000 == 0:
                    print(f"   📊 Επεξεργασία γραμμής {line_num:,}")
                    
        print(f"✅ Βρέθηκαν {len(eurlex_concepts):,} μοναδικά concepts στο EURLEX training")
        return eurlex_concepts
        
    except Exception as e:
        print(f"❌ Σφάλμα κατά τη φόρτωση: {e}")
        return set()

def create_label_to_concept_mapping(concepts_data: Dict[str, Any]) -> Dict[str, str]:
    """Δημιουργία mapping από label text σε concept ID"""
    print("🔄 Δημιουργία label-to-concept mapping...")
    
    label_mapping = {}
    
    for concept_id, concept_data in concepts_data.items():
        # Κύριο title
        title = concept_data.get('title', '').strip()
        if title:
            label_mapping[title] = concept_id
            
        # Alt labels
        alt_labels = concept_data.get('alt_labels', [])
        for alt_label in alt_labels:
            if alt_label and alt_label.strip():
                label_mapping[alt_label.strip()] = concept_id
                
        # EURLEX label αν υπάρχει
        eurlex_label = concept_data.get('eurlex_label', '').strip()
        if eurlex_label:
            label_mapping[eurlex_label] = concept_id
    
    print(f"✅ Δημιουργήθηκε mapping για {len(label_mapping):,} labels")
    return label_mapping

def filter_training_relevant_concepts(concepts_data: Dict[str, Any], 
                                     eurlex_concepts: set, 
                                     label_mapping: Dict[str, str]) -> Dict[str, Any]:
    """Φιλτράρισμα concepts που αναφέρονται στο EURLEX (με οποιονδήποτε τρόπο)"""
    print("🔄 Φιλτράρισμα concepts που αναφέρονται στο EURLEX...")
    
    filtered_concepts = {}
    found_via_mapping = set()
    
    # Βρίσκουμε όλα τα concept IDs που αντιστοιχούν στα EURLEX labels
    for eurlex_label in eurlex_concepts:
        if eurlex_label in label_mapping:
            concept_id = label_mapping[eurlex_label]
            found_via_mapping.add(concept_id)
    
    # Κρατάμε όλα τα concepts που βρέθηκαν
    for concept_id in found_via_mapping:
        if concept_id in concepts_data:
            filtered_concepts[concept_id] = concepts_data[concept_id]
    
    print(f"✅ Φιλτραρίστηκαν {len(filtered_concepts)} concepts από {len(concepts_data)}")
    print(f"📊 Αφαιρέθηκαν {len(concepts_data) - len(filtered_concepts)} concepts")
    print(f"🎯 Matched {len(found_via_mapping)} concepts via EURLEX labels")
    
    return filtered_concepts

def generate_filtered_statistics(original_data: Dict[str, Any], 
                                filtered_data: Dict[str, Any],
                                eurlex_concepts: set) -> Dict[str, Any]:
    """Δημιουργία statistics για το φιλτράρισμα"""
    print("📊 Δημιουργία statistics φιλτραρίσματος...")
    
    # Original stats
    original_with_alt_labels = sum(1 for v in original_data.values() if v.get('alt_labels'))
    original_with_parents = sum(1 for v in original_data.values() if v.get('parents'))
    original_with_eurlex = sum(1 for v in original_data.values() if v.get('eurlex_source'))
    original_with_paths = sum(1 for v in original_data.values() if v.get('hierarchical_paths'))
    
    # Filtered stats
    filtered_with_alt_labels = sum(1 for v in filtered_data.values() if v.get('alt_labels'))
    filtered_with_parents = sum(1 for v in filtered_data.values() if v.get('parents'))
    filtered_with_eurlex = sum(1 for v in filtered_data.values() if v.get('eurlex_source'))
    filtered_with_paths = sum(1 for v in filtered_data.values() if v.get('hierarchical_paths'))
    
    # EURLEX coverage analysis
    total_eurlex_labels = len(eurlex_concepts)
    
    stats = {
        "filtering_summary": {
            "original_concepts": len(original_data),
            "filtered_concepts": len(filtered_data),
            "removed_concepts": len(original_data) - len(filtered_data),
            "retention_percentage": round((len(filtered_data) / len(original_data)) * 100, 2)
        },
        "eurlex_coverage": {
            "total_eurlex_labels_in_training": total_eurlex_labels,
            "matched_concepts": len(filtered_data),
            "coverage_percentage": round((len(filtered_data) / total_eurlex_labels) * 100, 2) if total_eurlex_labels > 0 else 0
        },
        "content_comparison": {
            "original": {
                "concepts_with_alt_labels": original_with_alt_labels,
                "concepts_with_parents": original_with_parents,
                "concepts_with_eurlex_source": original_with_eurlex,
                "concepts_with_paths": original_with_paths
            },
            "filtered": {
                "concepts_with_alt_labels": filtered_with_alt_labels,
                "concepts_with_parents": filtered_with_parents,
                "concepts_with_eurlex_source": filtered_with_eurlex,
                "concepts_with_paths": filtered_with_paths
            }
        },
        "data_quality": {
            "concepts_with_complete_metadata": sum(1 for v in filtered_data.values() 
                                                 if v.get('alt_labels') and v.get('parents')),
            "concepts_with_any_enhancement": sum(1 for v in filtered_data.values() 
                                               if v.get('alt_labels') or v.get('parents') or v.get('hierarchical_paths')),
            "concepts_with_hierarchical_paths": filtered_with_paths,
            "average_alt_labels_per_concept": round(
                sum(len(v.get('alt_labels', [])) for v in filtered_data.values()) / len(filtered_data), 2
            ) if filtered_data else 0
        }
    }
    
    return stats

def save_filtered_concepts(filtered_concepts: Dict[str, Any], output_path: str):
    """Αποθήκευση των φιλτραρισμένων concepts"""
    print(f"💾 Αποθήκευση φιλτραρισμένων concepts στο {output_path}...")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_concepts, f, ensure_ascii=False, indent=2)
        print("✅ Επιτυχής αποθήκευση!")
    except Exception as e:
        print(f"❌ Σφάλμα κατά την αποθήκευση: {e}")

def main():
    """Κύρια συνάρτηση"""
    print("🚀 ΦΙΛΤΡΑΡΙΣΜΑ CONCEPTS - ΟΟΛΑ ΤΑ EURLEX CONCEPTS")
    print("=" * 60)
    
    # Διαδρομές αρχείων
    data_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(data_dir, "eurovoc_enhanced_concepts_with_paths.json")
    train_labels_path = os.path.join(data_dir, "eurlex57k_train_labels.jsonl")
    output_path = os.path.join(data_dir, "eurovoc_all_eurlex_concepts.json")
    stats_path = os.path.join(data_dir, "all_eurlex_filtering_statistics.json")
    
    # Φόρτωση enhanced concepts
    original_concepts = load_concepts_with_paths(input_path)
    if not original_concepts:
        print("❌ Δεν μπόρεσα να φορτώσω τα concepts")
        return
    
    # Φόρτωση EURLEX training labels
    eurlex_concepts = load_train_labels_concepts(train_labels_path)
    if not eurlex_concepts:
        print("❌ Δεν μπόρεσα να φορτώσω τα EURLEX concepts")
        return
    
    # Δημιουργία mapping
    label_mapping = create_label_to_concept_mapping(original_concepts)
    
    # Φιλτράρισμα
    filtered_concepts = filter_training_relevant_concepts(original_concepts, eurlex_concepts, label_mapping)
    
    if not filtered_concepts:
        print("❌ Δεν βρέθηκαν concepts με EURLEX references")
        return
    
    # Δημιουργία statistics
    stats = generate_filtered_statistics(original_concepts, filtered_concepts, eurlex_concepts)
    
    # Αποθήκευση
    save_filtered_concepts(filtered_concepts, output_path)
    
    # Αποθήκευση statistics
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print("\n📊 ΑΠΟΤΕΛΕΣΜΑΤΑ ΦΙΛΤΡΑΡΙΣΜΑΤΟΣ:")
    print(f"📁 Αρχικά concepts: {stats['filtering_summary']['original_concepts']:,}")
    print(f"✅ Φιλτραρισμένα concepts: {stats['filtering_summary']['filtered_concepts']:,}")
    print(f"❌ Αφαιρεμένα concepts: {stats['filtering_summary']['removed_concepts']:,}")
    print(f"📈 Ποσοστό διατήρησης: {stats['filtering_summary']['retention_percentage']}%")
    print(f"� EURLEX labels στο training: {stats['eurlex_coverage']['total_eurlex_labels_in_training']:,}")
    print(f"� Coverage EURLEX concepts: {stats['eurlex_coverage']['coverage_percentage']}%")
    print(f"🏷️  Concepts με alt_labels: {stats['content_comparison']['filtered']['concepts_with_alt_labels']}")
    print(f"🌳 Concepts με parents: {stats['content_comparison']['filtered']['concepts_with_parents']}")
    print(f"📚 Concepts με paths: {stats['content_comparison']['filtered']['concepts_with_paths']}")
    
    print(f"\n🎉 ΟΛΟΚΛΗΡΩΣΗ!")
    print(f"📁 Καθαρό αρχείο: {output_path}")
    print(f"📊 Statistics: {stats_path}")
    print("\n✨ Το νέο αρχείο περιέχει ΟΛΛΑ τα concepts που αναφέρονται στο EURLEX!")
    print("💡 Περιλαμβάνει concepts με ή χωρίς alt_labels/ιεραρχική δομή")

if __name__ == "__main__":
    main()
