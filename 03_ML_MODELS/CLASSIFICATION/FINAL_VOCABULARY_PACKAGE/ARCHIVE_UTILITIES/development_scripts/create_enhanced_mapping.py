#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Eurovision ID→Title Mapping Creator
============================================
Δημιουργεί εμπλουτισμένο JSON mapping με κατηγορίες και metadata.
"""

import csv
import json
from datetime import datetime

def create_enhanced_eurovoc_mapping():
    """
    Δημιουργεί εμπλουτισμένο Eurovision mapping με:
    - Title
    - Category code & name  
    - Redirect information
    - Preferred terms
    """
    
    print("🔧 ΔΗΜΙΟΥΡΓΙΑ ΕΜΠΛΟΥΤΙΣΜΕΝΟΥ EUROVISION MAPPING")
    print("="*60)
    
    csv_path = 'data/eurovoc_export_en.csv'
    output_path = 'data/eurovoc_enhanced_mapping.json'
    
    enhanced_mapping = {}
    categories_seen = set()
    
    print("📁 Φόρτωση Eurovision CSV...")
    
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f, delimiter=';')
        
        total_processed = 0
        
        for row in reader:
            total_processed += 1
            
            concept_id = row['ID']
            term = row['TERMS (PT-NPT)']
            relation = row['RELATIONS']
            preferred_term = row['PT']
            mt_category = row['MT']
            
            # Παράλειψη αν δεν έχει MT category
            if not mt_category:
                continue
                
            # Διαχωρισμός category code και name
            category_code = None
            category_name = None
            if ' ' in mt_category:
                parts = mt_category.split(' ', 1)
                category_code = parts[0]
                category_name = parts[1]
                categories_seen.add(mt_category)
            
            # Έλεγχος αν είναι redirect
            is_redirect = bool(relation == 'USE' and preferred_term)
            
            # Δημιουργία enhanced entry
            enhanced_entry = {
                'title': term,
                'category_code': category_code,
                'category_name': category_name,
                'category_full': mt_category,
                'is_redirect': is_redirect,
                'preferred_term': preferred_term if is_redirect else None,
                'relation_type': relation if relation else None
            }
            
            enhanced_mapping[concept_id] = enhanced_entry
            
            # Progress update
            if total_processed % 1000 == 0:
                print(f"  📊 Επεξεργάστηκαν {total_processed:,} εγγραφές...")
    
    print(f"✅ Ολοκληρώθηκε! Επεξεργάστηκαν {total_processed:,} εγγραφές")
    print(f"📦 Δημιουργήθηκαν {len(enhanced_mapping):,} enhanced mappings")
    print(f"🏷️  Βρέθηκαν {len(categories_seen):,} διαφορετικές κατηγορίες")
    
    # Στατιστικά
    redirects_count = sum(1 for entry in enhanced_mapping.values() if entry['is_redirect'])
    direct_terms = len(enhanced_mapping) - redirects_count
    
    print(f"\n📊 ΣΤΑΤΙΣΤΙΚΑ MAPPING:")
    print("-" * 40)
    print(f"Άμεσοι όροι:      {direct_terms:,} ({direct_terms/len(enhanced_mapping)*100:.1f}%)")
    print(f"Redirects:        {redirects_count:,} ({redirects_count/len(enhanced_mapping)*100:.1f}%)")
    
    # Τοπ κατηγορίες
    category_counts = {}
    for entry in enhanced_mapping.values():
        if entry['category_full']:
            category_counts[entry['category_full']] = category_counts.get(entry['category_full'], 0) + 1
    
    print(f"\n🏆 ΤΟΠ 10 ΚΑΤΗΓΟΡΙΕΣ:")
    print("-" * 60)
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{count:>4,} όροι: {category}")
    
    # Δημιουργία metadata
    metadata = {
        'creation_info': {
            'created_at': datetime.now().isoformat(),
            'source_file': 'eurovoc_export_en.csv',
            'total_concepts': len(enhanced_mapping),
            'total_categories': len(categories_seen)
        },
        'statistics': {
            'direct_terms': direct_terms,
            'redirect_terms': redirects_count,
            'top_categories': dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        },
        'structure_info': {
            'description': 'Enhanced Eurovision concept mapping with categories and redirects',
            'fields': {
                'title': 'The term/concept name',
                'category_code': 'Numeric category code (e.g., 7211)',
                'category_name': 'Category description (e.g., regions of EU Member States)',
                'category_full': 'Full category string (code + name)',
                'is_redirect': 'True if this is a redirect to another term',
                'preferred_term': 'Target term if this is a redirect',
                'relation_type': 'Type of relation (usually USE for redirects)'
            }
        }
    }
    
    # Τελικό JSON
    final_output = {
        'metadata': metadata,
        'concepts': enhanced_mapping
    }
    
    # Αποθήκευση
    print(f"\n💾 Αποθήκευση εμπλουτισμένου mapping...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    # Δημιουργία συμπαγούς έκδοσης (μόνο concepts)
    compact_path = 'data/eurovoc_enhanced_concepts_only.json'
    with open(compact_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Αποθηκεύτηκε στο: {output_path}")
    print(f"✅ Συμπαγής έκδοση: {compact_path}")
    
    # Sample entries για demonstration
    sample_path = 'data/eurovoc_enhanced_sample.json'
    sample_concepts = dict(list(enhanced_mapping.items())[:20])
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_concepts, f, ensure_ascii=False, indent=2)
    
    print(f"📋 Δείγμα αποθηκεύτηκε: {sample_path}")
    
    print(f"\n🎯 ΠΑΡΑΔΕΙΓΜΑΤΑ ΕΜΠΛΟΥΤΙΣΜΕΝΩΝ ENTRIES:")
    print("-" * 60)
    
    # Εμφάνιση samples
    count = 0
    for concept_id, entry in enhanced_mapping.items():
        if count >= 5:
            break
            
        print(f"📍 {concept_id}:")
        print(f"   Title: \"{entry['title']}\"")
        print(f"   Category: {entry['category_full']}")
        if entry['is_redirect']:
            print(f"   Redirect to: \"{entry['preferred_term']}\"")
        print()
        count += 1
    
    return final_output

if __name__ == "__main__":
    result = create_enhanced_eurovoc_mapping()
