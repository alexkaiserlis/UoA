#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eurovision Concepts ID→Title Extractor
=====================================

Δημιουργεί ένα καθαρό JSON αρχείο με Eurovision concept mappings
από το επίσημο eurovoc_export_en.csv αρχείο.

Output format:
{
  "Eurovision_ID": "Human_Readable_Title",
  ...
}
"""

import json
import csv
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eurovoc_id_title_extraction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def extract_eurovoc_mappings(csv_file: str) -> dict:
    """
    Εξάγει Eurovision ID→Title mappings από το CSV αρχείο.
    
    Args:
        csv_file: Το path του eurovoc_export_en.csv αρχείου
        
    Returns:
        Dictionary με Eurovision ID → Title mappings
    """
    mappings = {}
    skipped_entries = 0
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            # Skip BOM if present
            content = file.read()
            if content.startswith('\ufeff'):
                content = content[1:]
            
            reader = csv.DictReader(content.splitlines(), delimiter=';')
            
            for row_num, row in enumerate(reader, 1):
                eurovoc_id = row['ID'].strip()
                
                # Επιλογή τίτλου με προτεραιότητα
                title = ""
                
                # 1. Προτεραιότητα στο TERMS field
                if 'TERMS (PT-NPT)' in row and row['TERMS (PT-NPT)'].strip():
                    title = row['TERMS (PT-NPT)'].strip()
                
                # 2. Fallback στο PT field
                elif 'PT' in row and row['PT'].strip():
                    title = row['PT'].strip()
                
                # 3. Έλεγχος αν είναι USE reference
                if 'RELATIONS' in row and row['RELATIONS'].strip() == 'USE':
                    # Αν είναι USE reference, χρησιμοποιούμε το PT ως τίτλο
                    if 'PT' in row and row['PT'].strip():
                        title = row['PT'].strip()
                    else:
                        title = f"See: {row.get('TERMS (PT-NPT)', '').strip()}"
                
                # 4. Τελικό fallback
                if not title:
                    title = f"Concept {eurovoc_id}"
                    skipped_entries += 1
                
                # Καθάρισμα τίτλου
                title = title.replace('\n', ' ').replace('\r', ' ').strip()
                
                # Αποθήκευση mapping
                mappings[eurovoc_id] = title
                
        logging.info(f"Επεξεργάστηκαν {len(mappings)} Eurovision concepts")
        if skipped_entries > 0:
            logging.warning(f"Δημιουργήθηκαν αυτόματοι τίτλοι για {skipped_entries} concepts")
        
    except Exception as e:
        logging.error(f"Σφάλμα κατά την ανάγνωση του CSV: {e}")
        raise
    
    return mappings

def analyze_mappings(mappings: dict) -> dict:
    """
    Αναλύει τα mappings και δημιουργεί στατιστικά.
    
    Args:
        mappings: Τα Eurovision ID→Title mappings
        
    Returns:
        Στατιστικά των mappings
    """
    # Κατηγοριοποίηση concepts
    categories = {
        'numeric_ids': [],          # Αριθμητικά IDs
        'alphanumeric_ids': [],     # Αλφαριθμητικά IDs (c_xxxx)
        'auto_generated_titles': [], # Αυτόματα titles
        'use_references': [],       # USE references
        'regular_concepts': []      # Κανονικά concepts
    }
    
    for concept_id, title in mappings.items():
        # Κατηγοριοποίηση ID
        if concept_id.isdigit():
            categories['numeric_ids'].append(concept_id)
        elif concept_id.startswith('c_'):
            categories['alphanumeric_ids'].append(concept_id)
        
        # Κατηγοριοποίηση τίτλου
        if title.startswith('Concept '):
            categories['auto_generated_titles'].append(concept_id)
        elif title.startswith('See: '):
            categories['use_references'].append(concept_id)
        else:
            categories['regular_concepts'].append(concept_id)
    
    # Μήκη τίτλων
    title_lengths = [len(title) for title in mappings.values()]
    
    # Πιο συχνές λέξεις στους τίτλους
    word_frequency = {}
    for title in mappings.values():
        words = title.lower().split()
        for word in words:
            if len(word) > 3:  # Αγνοούμε πολύ μικρές λέξεις
                word_frequency[word] = word_frequency.get(word, 0) + 1
    
    most_common_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
    
    stats = {
        'total_concepts': len(mappings),
        'categories': {k: len(v) for k, v in categories.items()},
        'sample_categories': {k: v[:5] for k, v in categories.items()},
        'title_length_stats': {
            'min': min(title_lengths) if title_lengths else 0,
            'max': max(title_lengths) if title_lengths else 0,
            'avg': sum(title_lengths) / len(title_lengths) if title_lengths else 0
        },
        'most_common_words': most_common_words
    }
    
    return stats

def save_mappings_json(mappings: dict, output_file: str):
    """
    Αποθηκεύει τα mappings σε JSON format.
    
    Args:
        mappings: Τα Eurovision ID→Title mappings
        output_file: Το path του output αρχείου
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(mappings, file, ensure_ascii=False, indent=2, sort_keys=True)
        
        logging.info(f"Eurovision mappings αποθηκεύτηκαν στο: {output_file}")
        
    except Exception as e:
        logging.error(f"Σφάλμα κατά την αποθήκευση: {e}")
        raise

def create_sample_file(mappings: dict, sample_file: str, sample_size: int = 50):
    """
    Δημιουργεί ένα sample file για επίδειξη.
    
    Args:
        mappings: Τα Eurovision ID→Title mappings
        sample_file: Το path του sample αρχείου
        sample_size: Πόσα entries να συμπεριλάβει
    """
    # Παίρνουμε ένα ισορροπημένο sample
    numeric_ids = [k for k in mappings.keys() if k.isdigit()]
    alphanumeric_ids = [k for k in mappings.keys() if k.startswith('c_')]
    
    sample_numeric = numeric_ids[:sample_size//2]
    sample_alpha = alphanumeric_ids[:sample_size//2]
    
    sample_keys = sample_numeric + sample_alpha
    sample_mappings = {k: mappings[k] for k in sample_keys if k in mappings}
    
    try:
        with open(sample_file, 'w', encoding='utf-8') as file:
            json.dump(sample_mappings, file, ensure_ascii=False, indent=2, sort_keys=True)
        
        logging.info(f"Sample mappings αποθηκεύτηκαν στο: {sample_file}")
        
    except Exception as e:
        logging.error(f"Σφάλμα κατά την αποθήκευση του sample: {e}")

def main():
    """Κύρια συνάρτηση του script."""
    
    # Paths των αρχείων
    csv_file = 'eurovoc_export_en.csv'
    output_file = 'eurovoc_id_title_mappings.json'
    sample_file = 'eurovoc_id_title_mappings_sample.json'
    stats_file = 'eurovoc_mappings_statistics.json'
    
    logging.info("Ξεκινάει η εξαγωγή Eurovision ID→Title mappings...")
    
    try:
        # 1. Εξαγωγή mappings από CSV
        logging.info("Ανάγνωση Eurovision concepts από CSV...")
        mappings = extract_eurovoc_mappings(csv_file)
        
        # 2. Ανάλυση mappings
        logging.info("Ανάλυση mappings...")
        stats = analyze_mappings(mappings)
        
        # 3. Αποθήκευση πλήρων mappings
        logging.info("Αποθήκευση πλήρων mappings...")
        save_mappings_json(mappings, output_file)
        
        # 4. Δημιουργία sample
        logging.info("Δημιουργία sample file...")
        create_sample_file(mappings, sample_file)
        
        # 5. Αποθήκευση στατιστικών
        logging.info("Αποθήκευση στατιστικών...")
        with open(stats_file, 'w', encoding='utf-8') as file:
            json.dump(stats, file, ensure_ascii=False, indent=2)
        
        # 6. Εμφάνιση αποτελεσμάτων
        print("\n" + "="*60)
        print("EUROVISION ID→TITLE MAPPINGS EXTRACTION")
        print("="*60)
        print(f"Συνολικά concepts: {stats['total_concepts']:,}")
        
        print(f"\nΚατηγορίες:")
        for category, count in stats['categories'].items():
            print(f"  {category}: {count:,}")
        
        print(f"\nΤίτλοι:")
        print(f"  Μέσο μήκος: {stats['title_length_stats']['avg']:.1f} χαρακτήρες")
        print(f"  Εύρος: {stats['title_length_stats']['min']}-{stats['title_length_stats']['max']} χαρακτήρες")
        
        print(f"\nΠιο συχνές λέξεις στους τίτλους:")
        for word, freq in stats['most_common_words'][:5]:
            print(f"  '{word}': {freq} φορές")
        
        print(f"\nΠαραδείγματα mappings:")
        sample_keys = list(mappings.keys())[:3]
        for key in sample_keys:
            print(f"  {key}: '{mappings[key]}'")
        
        print(f"\nΑρχεία που δημιουργήθηκαν:")
        print(f"  - {output_file} (πλήρες mapping)")
        print(f"  - {sample_file} (sample 50 entries)")
        print(f"  - {stats_file} (στατιστικά)")
        print(f"  - eurovoc_id_title_extraction.log (log αρχείο)")
        
        logging.info("Επιτυχής ολοκλήρωση της εξαγωγής!")
        
    except Exception as e:
        logging.error(f"Σφάλμα κατά την εκτέλεση: {e}")
        raise

if __name__ == "__main__":
    main()
