#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script για προσθήκη ιεραρχικών διαδρομών από EURLEX57K train labels
στο enhanced concepts αρχείο
"""

import json
import os
from collections import defaultdict
from typing import Dict, Any, List, Set

def load_enhanced_concepts(file_path: str) -> Dict[str, Any]:
    """Φόρτωση του enhanced concepts αρχείου"""
    print("🔄 Φόρτωση enhanced concepts...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Φορτώθηκαν {len(data)} concepts")
        return data
    except Exception as e:
        print(f"❌ Σφάλμα κατά τη φόρτωση: {e}")
        return {}

def load_train_labels(file_path: str) -> List[Dict[str, Any]]:
    """Φόρτωση των train labels από JSONL"""
    print("🔄 Φόρτωση train labels...")
    
    labels = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    label_data = json.loads(line.strip())
                    labels.append(label_data)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Σφάλμα στη γραμμή {line_num}: {e}")
                    continue
                    
                # Progress indicator για μεγάλα αρχεία
                if line_num % 50000 == 0:
                    print(f"   📊 Επεξεργασία γραμμής {line_num:,}")
                    
        print(f"✅ Φορτώθηκαν {len(labels):,} label entries")
        return labels
        
    except Exception as e:
        print(f"❌ Σφάλμα κατά τη φόρτωση train labels: {e}")
        return []

def create_label_to_concept_mapping(enhanced_concepts: Dict[str, Any]) -> Dict[str, str]:
    """Δημιουργία mapping από label text σε concept ID"""
    print("🔄 Δημιουργία label-to-concept mapping...")
    
    label_mapping = {}
    
    for concept_id, concept_data in enhanced_concepts.items():
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

def build_hierarchical_paths(train_labels: List[Dict[str, Any]], 
                           label_mapping: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    """Κατασκευή ιεραρχικών διαδρομών για κάθε concept"""
    print("🔄 Κατασκευή ιεραρχικών διαδρομών...")
    
    concept_paths = defaultdict(list)
    unmatched_labels = set()
    
    for entry in train_labels:
        # Εξαγωγή path από levels
        path_levels = []
        for level in ['level_1_label', 'level_2_label', 'level_3_label', 'level_4_label', 'level_5_label']:
            label = entry.get(level, '').strip()
            if label:
                path_levels.append(label)
            else:
                break  # Σταματάμε στο πρώτο κενό level
        
        if not path_levels:
            continue
            
        # Βρίσκουμε το concept ID για το τελευταίο (leaf) label
        leaf_label = path_levels[-1]
        concept_id = label_mapping.get(leaf_label)
        
        if concept_id:
            # Δημιουργία path object
            path_obj = {
                "path": path_levels,
                "depth": len(path_levels),
                "root_category": path_levels[0],
                "leaf_concept": leaf_label,
                "celex_id": entry.get('celex_id', ''),
                "full_path_string": " > ".join(path_levels)
            }
            
            concept_paths[concept_id].append(path_obj)
        else:
            unmatched_labels.add(leaf_label)
    
    print(f"✅ Δημιουργήθηκαν διαδρομές για {len(concept_paths):,} concepts")
    print(f"⚠️  {len(unmatched_labels):,} unmatched labels")
    
    return dict(concept_paths)

def calculate_path_statistics(paths: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Υπολογισμός statistics για τις διαδρομές ενός concept"""
    if not paths:
        return {}
    
    depths = [p['depth'] for p in paths]
    root_categories = [p['root_category'] for p in paths]
    celex_ids = [p['celex_id'] for p in paths if p['celex_id']]
    
    return {
        "total_paths": len(paths),
        "unique_paths": len(set(p['full_path_string'] for p in paths)),
        "depth_stats": {
            "min_depth": min(depths),
            "max_depth": max(depths),
            "avg_depth": round(sum(depths) / len(depths), 2)
        },
        "root_categories": {
            "unique_roots": list(set(root_categories)),
            "root_count": len(set(root_categories))
        },
        "document_coverage": {
            "unique_documents": len(set(celex_ids)),
            "total_document_mentions": len(celex_ids)
        }
    }

def enhance_concepts_with_paths(enhanced_concepts: Dict[str, Any], 
                              concept_paths: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Προσθήκη ιεραρχικών διαδρομών στα concepts"""
    print("🔄 Προσθήκη ιεραρχικών διαδρομών στα concepts...")
    
    enhanced_count = 0
    
    for concept_id, concept_data in enhanced_concepts.items():
        if concept_id in concept_paths:
            paths = concept_paths[concept_id]
            
            # Αφαίρεση duplicates
            unique_paths = []
            seen_paths = set()
            for path in paths:
                path_key = path['full_path_string']
                if path_key not in seen_paths:
                    unique_paths.append(path)
                    seen_paths.add(path_key)
            
            # Προσθήκη στο concept
            concept_data['hierarchical_paths'] = unique_paths
            concept_data['path_metadata'] = calculate_path_statistics(unique_paths)
            
            enhanced_count += 1
    
    print(f"✅ Προστέθηκαν διαδρομές σε {enhanced_count:,} concepts")
    return enhanced_concepts

def save_enhanced_concepts_with_paths(enhanced_concepts: Dict[str, Any], output_path: str):
    """Αποθήκευση των enhanced concepts με paths"""
    print(f"💾 Αποθήκευση enhanced concepts με paths στο {output_path}...")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_concepts, f, ensure_ascii=False, indent=2)
        print("✅ Επιτυχής αποθήκευση!")
    except Exception as e:
        print(f"❌ Σφάλμα κατά την αποθήκευση: {e}")

def generate_path_statistics_report(enhanced_concepts: Dict[str, Any]) -> Dict[str, Any]:
    """Δημιουργία συνολικού report statistics"""
    print("📊 Δημιουργία statistics report...")
    
    concepts_with_paths = 0
    total_paths = 0
    all_depths = []
    all_root_categories = set()
    all_documents = set()
    
    for concept_id, concept_data in enhanced_concepts.items():
        if 'hierarchical_paths' in concept_data:
            concepts_with_paths += 1
            paths = concept_data['hierarchical_paths']
            total_paths += len(paths)
            
            for path in paths:
                all_depths.append(path['depth'])
                all_root_categories.add(path['root_category'])
                if path['celex_id']:
                    all_documents.add(path['celex_id'])
    
    report = {
        "concepts_overview": {
            "total_concepts": len(enhanced_concepts),
            "concepts_with_paths": concepts_with_paths,
            "concepts_without_paths": len(enhanced_concepts) - concepts_with_paths,
            "path_coverage_percentage": round((concepts_with_paths / len(enhanced_concepts)) * 100, 2)
        },
        "paths_overview": {
            "total_hierarchical_paths": total_paths,
            "average_paths_per_concept": round(total_paths / concepts_with_paths, 2) if concepts_with_paths > 0 else 0
        },
        "depth_analysis": {
            "min_depth": min(all_depths) if all_depths else 0,
            "max_depth": max(all_depths) if all_depths else 0,
            "avg_depth": round(sum(all_depths) / len(all_depths), 2) if all_depths else 0,
            "depth_distribution": {str(d): all_depths.count(d) for d in set(all_depths)}
        },
        "category_analysis": {
            "unique_root_categories": len(all_root_categories),
            "root_categories_list": sorted(list(all_root_categories))
        },
        "document_analysis": {
            "unique_source_documents": len(all_documents),
            "total_document_references": len([path['celex_id'] for concept_data in enhanced_concepts.values() 
                                           if 'hierarchical_paths' in concept_data 
                                           for path in concept_data['hierarchical_paths'] 
                                           if path['celex_id']])
        }
    }
    
    return report

def main():
    """Κύρια συνάρτηση"""
    print("🚀 ΠΡΟΣΘΗΚΗ ΙΕΡΑΡΧΙΚΩΝ ΔΙΑΔΡΟΜΩΝ ΣΤΑ CONCEPTS")
    print("=" * 60)
    
    # Διαδρομές αρχείων
    data_dir = os.path.dirname(os.path.abspath(__file__))
    enhanced_concepts_path = os.path.join(data_dir, "eurovoc_enhanced_concepts_with_eurlex.json")
    train_labels_path = os.path.join(data_dir, "eurlex57k_train_labels.jsonl")
    output_path = os.path.join(data_dir, "eurovoc_enhanced_concepts_with_paths.json")
    stats_path = os.path.join(data_dir, "hierarchical_paths_statistics.json")
    
    # Φόρτωση δεδομένων
    enhanced_concepts = load_enhanced_concepts(enhanced_concepts_path)
    if not enhanced_concepts:
        print("❌ Δεν μπόρεσα να φορτώσω τα enhanced concepts")
        return
    
    train_labels = load_train_labels(train_labels_path)
    if not train_labels:
        print("❌ Δεν μπόρεσα να φορτώσω τα train labels")
        return
    
    # Δημιουργία mappings και paths
    label_mapping = create_label_to_concept_mapping(enhanced_concepts)
    concept_paths = build_hierarchical_paths(train_labels, label_mapping)
    
    # Επέκταση concepts με paths
    enhanced_concepts_with_paths = enhance_concepts_with_paths(enhanced_concepts, concept_paths)
    
    # Δημιουργία statistics report
    stats_report = generate_path_statistics_report(enhanced_concepts_with_paths)
    
    # Αποθήκευση
    save_enhanced_concepts_with_paths(enhanced_concepts_with_paths, output_path)
    
    # Αποθήκευση statistics
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_report, f, ensure_ascii=False, indent=2)
    
    print("\n📊 ΤΕΛΙΚΑ STATISTICS:")
    print(f"📁 Concepts με paths: {stats_report['concepts_overview']['concepts_with_paths']:,}")
    print(f"📈 Coverage: {stats_report['concepts_overview']['path_coverage_percentage']}%")
    print(f"🌳 Συνολικές διαδρομές: {stats_report['paths_overview']['total_hierarchical_paths']:,}")
    print(f"📚 Root categories: {stats_report['category_analysis']['unique_root_categories']}")
    print(f"📄 Source documents: {stats_report['document_analysis']['unique_source_documents']:,}")
    
    print(f"\n🎉 ΟΛΟΚΛΗΡΩΣΗ!")
    print(f"📁 Enhanced concepts: {output_path}")
    print(f"📊 Statistics report: {stats_path}")

if __name__ == "__main__":
    main()
