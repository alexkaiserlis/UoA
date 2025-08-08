#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eurovision CSV Analysis Tool
============================
Αναλύει τη δομή του Eurovision CSV αρχείου για να κατανοήσουμε τις στήλες.
"""

import csv
import json
from collections import Counter, defaultdict

def analyze_eurovoc_csv():
    """Αναλύει το Eurovision CSV αρχείο και τις στήλες του."""
    
    print("📊 ΑΝΑΛΥΣΗ EUROVISION CSV ΑΡΧΕΙΟΥ")
    print("="*50)
    
    csv_path = 'data/eurovoc_export_en.csv'
    
    # Στατιστικά για κάθε στήλη
    relations_stats = Counter()
    mt_codes = Counter()
    pt_examples = []
    
    total_rows = 0
    rows_with_relations = 0
    rows_with_pt = 0
    rows_with_mt = 0
    
    print("📁 Φόρτωση και ανάλυση CSV...")
    
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f, delimiter=';')
        
        # Εμφάνιση headers
        headers = reader.fieldnames
        print(f"\n📋 ΣΤΗΛΕΣ CSV:")
        print("-" * 30)
        for i, header in enumerate(headers, 1):
            print(f"{i}. {header}")
        
        for row in reader:
            total_rows += 1
            
            # Ανάλυση RELATIONS στήλης
            if row['RELATIONS']:
                relations_stats[row['RELATIONS']] += 1
                rows_with_relations += 1
            
            # Ανάλυση PT στήλης (Preferred Term)
            if row['PT']:
                pt_examples.append(row['PT'])
                rows_with_pt += 1
            
            # Ανάλυση MT στήλης (Microthesaurus)
            if row['MT']:
                mt_codes[row['MT']] += 1
                rows_with_mt += 1
            
            # Σταμάτα στις πρώτες 1000 γραμμές για γρήγορη ανάλυση
            if total_rows >= 1000:
                break
    
    print(f"\n📊 ΓΕΝΙΚΑ ΣΤΑΤΙΣΤΙΚΑ (πρώτες {total_rows:,} γραμμές):")
    print("-" * 50)
    print(f"Συνολικές γραμμές:     {total_rows:,}")
    print(f"Με RELATIONS:          {rows_with_relations:,} ({rows_with_relations/total_rows*100:.1f}%)")
    print(f"Με PT (preferred):     {rows_with_pt:,} ({rows_with_pt/total_rows*100:.1f}%)")
    print(f"Με MT (category):      {rows_with_mt:,} ({rows_with_mt/total_rows*100:.1f}%)")
    
    print(f"\n🔗 ΑΝΑΛΥΣΗ RELATIONS ΣΤΗΛΗΣ:")
    print("-" * 50)
    print("Η στήλη RELATIONS δείχνει σχέσεις μεταξύ όρων:")
    if relations_stats:
        for relation, count in relations_stats.most_common(10):
            print(f"  • {relation}: {count:,} φορές")
    else:
        print("  • Κενή στις περισσότερες γραμμές")
    
    print(f"\n📝 ΑΝΑΛΥΣΗ PT ΣΤΗΛΗΣ (Preferred Terms):")
    print("-" * 50)
    print("Η στήλη PT περιέχει τον κύριο όρο όταν υπάρχει redirect:")
    if pt_examples:
        print("Παραδείγματα:")
        for example in pt_examples[:10]:
            print(f"  • \"{example}\"")
    
    print(f"\n🏷️  ΑΝΑΛΥΣΗ MT ΣΤΗΛΗΣ (Microthesaurus - Κατηγορίες):")
    print("-" * 50)
    print("Η στήλη MT περιέχει κωδικούς κατηγοριών με περιγραφές:")
    if mt_codes:
        print("Τοπ κατηγορίες:")
        for mt_code, count in mt_codes.most_common(10):
            print(f"  • {mt_code}: {count:,} όροι")
    
    # Δημιουργία λεξικού κατηγοριών
    categories_dict = {}
    if mt_codes:
        for mt_code in mt_codes.keys():
            if ' ' in mt_code:
                code_part = mt_code.split(' ')[0]
                description_part = ' '.join(mt_code.split(' ')[1:])
                categories_dict[code_part] = description_part
    
    print(f"\n💡 ΣΥΜΠΕΡΑΣΜΑΤΑ:")
    print("-" * 50)
    print("📋 ΣΤΗΛΕΣ ΚΑΙ ΠΕΡΙΕΧΟΜΕΝΟ:")
    print("  • ID: Μοναδικός αναγνωριστικός κωδικός")
    print("  • TERMS (PT-NPT): Ο όρος (κύριος ή εναλλακτικός)")
    print("  • RELATIONS: Σχέσεις όπως 'USE' (συνώνυμα/redirects)")
    print("  • PT: Preferred Term - ο κύριος όρος όταν υπάρχει redirect")
    print("  • MT: Microthesaurus - κατηγορία θεματολογίας (πολύ χρήσιμο!)")
    
    print(f"\n🎯 ΣΥΣΤΑΣΗ ΓΙΑ ΕΜΠΛΟΥΤΙΣΜΟ:")
    print("-" * 50)
    print("✅ Η στήλη MT είναι ΕΞΑΙΡΕΤΙΚΑ χρήσιμη!")
    print("   • Δίνει θεματική κατηγοριοποίηση σε κάθε concept")
    print("   • Περιέχει αριθμό κατηγορίας + περιγραφή")
    print("   • Παραδείγματα:")
    if mt_codes:
        examples = list(mt_codes.keys())[:5]
        for example in examples:
            print(f"     - {example}")
    
    print(f"\n📦 ΠΡΟΤΕΙΝΟΜΕΝΗ ΔΟΜΗ ΓΙΑ ΝΕΟ JSON:")
    print("-" * 50)
    print("Προτείνω να δημιουργήσουμε εμπλουτισμένο mapping:")
    print('''{
  "concept_id": {
    "title": "όνομα concept",
    "category_code": "κωδικός θεματολογίας",  
    "category_name": "όνομα θεματολογίας",
    "is_redirect": true/false,
    "preferred_term": "κύριος όρος αν υπάρχει redirect"
  }
}''')
    
    return {
        'total_analyzed': total_rows,
        'relations_stats': dict(relations_stats),
        'categories_dict': categories_dict,
        'mt_stats': dict(mt_codes.most_common(20))
    }

if __name__ == "__main__":
    results = analyze_eurovoc_csv()
