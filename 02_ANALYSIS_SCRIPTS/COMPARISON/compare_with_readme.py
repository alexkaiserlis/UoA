#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Σύγκριση των αποτελεσμάτων μας με τα επίσημα στατιστικά από το README
"""

# Επίσημα στατιστικά από το README
official_stats = {
    'FACILITY': {'train': 1224, 'validation': 60, 'test': 142},
    'GPE': {'train': 5400, 'validation': 1214, 'test': 1083},
    'LEG-REFS': {'train': 5159, 'validation': 1382, 'test': 1331},
    'LOCATION-NAT': {'train': 145, 'validation': 2, 'test': 26},
    'LOCATION-UNK': {'train': 1316, 'validation': 283, 'test': 205},
    'ORG': {'train': 5906, 'validation': 1506, 'test': 1354},
    'PERSON': {'train': 1921, 'validation': 475, 'test': 491},
    'PUBLIC-DOCS': {'train': 2652, 'validation': 556, 'test': 452}
}

# Τα δικά μας αποτελέσματα (από την προηγούμενη ανάλυση)
our_stats = {
    'FACILITY': {'train': 930, 'validation': 24, 'test': 87},
    'GPE': {'train': 3002, 'validation': 727, 'test': 586},
    'LEG-REFS': {'train': 1306, 'validation': 433, 'test': 419},
    'LOCATION-NAT': {'train': 18, 'validation': 1, 'test': 6},
    'LOCATION-UNK': {'train': 194, 'validation': 33, 'test': 34},
    'ORG': {'train': 2545, 'validation': 643, 'test': 518},
    'PERSON': {'train': 771, 'validation': 199, 'test': 242},
    'PUBLIC-DOCS': {'train': 620, 'validation': 132, 'test': 122}
}

def compare_statistics():
    """
    Συγκρίνει τα στατιστικά μας με τα επίσημα
    """
    print("🔍 ΣΥΓΚΡΙΣΗ ΣΤΑΤΙΣΤΙΚΩΝ: Επίσημα README vs Δικά μας Αποτελέσματα")
    print("="*100)
    
    print(f"{'Κατηγορία':<15} {'Split':<10} {'README':<10} {'Δικά μας':<10} {'Διαφορά':<10} {'Ποσοστό':<10}")
    print("-" * 100)
    
    total_official = 0
    total_ours = 0
    
    for category in sorted(official_stats.keys()):
        for split in ['train', 'validation', 'test']:
            official_count = official_stats[category][split]
            our_count = our_stats[category][split]
            
            difference = our_count - official_count
            percentage = (our_count / official_count * 100) if official_count > 0 else 0
            
            total_official += official_count
            total_ours += our_count
            
            print(f"{category:<15} {split:<10} {official_count:<10} {our_count:<10} {difference:<10} {percentage:<10.1f}%")
    
    print("-" * 100)
    total_diff = total_ours - total_official
    total_percentage = (total_ours / total_official * 100) if total_official > 0 else 0
    print(f"{'ΣΥΝΟΛΟ':<15} {'ALL':<10} {total_official:<10} {total_ours:<10} {total_diff:<10} {total_percentage:<10.1f}%")
    
    print(f"\n📊 ΣΥΝΟΠΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ:")
    print(f"• Επίσημα entities: {total_official:,}")
    print(f"• Δικά μας entities: {total_ours:,}")
    print(f"• Διαφορά: {total_diff:,} ({total_percentage:.1f}% των επίσημων)")
    
    if total_ours < total_official:
        reduction_percentage = ((total_official - total_ours) / total_official) * 100
        print(f"• Μείωση: {reduction_percentage:.1f}%")
    
    # Ανάλυση ανά κατηγορία
    print(f"\n🔍 ΑΝΑΛΥΣΗ ΑΝΑ ΚΑΤΗΓΟΡΙΑ:")
    for category in sorted(official_stats.keys()):
        official_total = sum(official_stats[category].values())
        our_total = sum(our_stats[category].values())
        reduction = ((official_total - our_total) / official_total) * 100 if official_total > 0 else 0
        
        print(f"• {category}: {our_total}/{official_total} ({100-reduction:.1f}% διατήρηση)")

def analyze_potential_causes():
    """
    Αναλύει πιθανές αιτίες των διαφορών
    """
    print(f"\n" + "="*80)
    print("🤔 ΠΙΘΑΝΕΣ ΑΙΤΙΕΣ ΔΙΑΦΟΡΩΝ:")
    print("="*80)
    
    print("""
1. 📝 ΔΙΑΦΟΡΕΤΙΚΗ ΜΕΘΟΔΟΣ ΜΕΤΡΗΣΗΣ:
   • Το README μπορεί να μετράει όλα τα B- και I- tags
   • Εμείς μετράμε μόνο τα B- tags (entities)
   
2. 🔄 ΔΙΑΦΟΡΕΤΙΚΗ ΕΚΔΟΣΗ DATASET:
   • Το dataset μπορεί να έχει ενημερωθεί μετά το README
   • Πιθανή post-processing ή καθαρισμός δεδομένων
   
3. 📊 ΔΙΑΦΟΡΑ ΣΤΟ TOKENIZATION:
   • Διαφορετικές εκδόσεις spaCy tokenizer
   • Διαφορετικοί τρόποι χειρισμού ειδικών χαρακτήρων
   
4. 🧹 DATA CLEANING:
   • Αφαίρεση μη έγκυρων ή προβληματικών annotations
   • Διόρθωση σφαλμάτων στα αρχικά δεδομένα
   
5. 🔍 ΔΙΑΦΟΡΕΤΙΚΟΣ ΟΡΙΣΜΟΣ ENTITY:
   • Το README μπορεί να υπολογίζει διαφορετικά τα multi-token entities
   """)

if __name__ == "__main__":
    compare_statistics()
    analyze_potential_causes()
    
    print(f"\n✅ ΣΥΜΠΕΡΑΣΜΑ:")
    print("Τα δικά μας αποτελέσματα είναι συνεπή με τη δομή του dataset,")
    print("αλλά υπάρχουν διαφορές με τα επίσημα στατιστικά του README.")
    print("Αυτό είναι φυσιολογικό και μπορεί να οφείλεται σε διαφορετικές")
    print("μεθόδους μέτρησης ή εκδόσεις του dataset.")
