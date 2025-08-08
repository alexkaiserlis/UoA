"""
Comparative Greek Legal NER Dataset Analysis Script

Αναλύει και συγκρίνει δύο datasets:
1. joelniklaus/greek_legal_ner 
2. joelito/lextreme (greek_legal_ner config)

Μετράει:
- Πόσα documents υπάρχουν σε κάθε split
- Πόσα instances υπάρχουν από κάθε κατηγορία IOB
- Συγκριτική ανάλυση μεταξύ των δύο datasets

Αποτελέσματα: Αποθηκεύονται σε txt αρχεία
"""

import os
from datasets import load_dataset
from collections import Counter, defaultdict
from datetime import datetime


def get_iob_label_mapping():
    """
    Επιστρέφει mapping από αριθμούς σε IOB labels για το joelito/lextreme dataset
    Βασισμένο στην πραγματική αντιστοιχία που βρέθηκε από τα datasets
    """
    return {
        0: 'O',
        1: 'B-ORG',
        2: 'I-ORG', 
        3: 'B-GPE',
        4: 'I-GPE',
        5: 'B-LEG-REFS',
        6: 'I-LEG-REFS',
        7: 'B-PUBLIC-DOCS',
        8: 'I-PUBLIC-DOCS',
        9: 'B-PERSON',
        10: 'I-PERSON',
        11: 'B-FACILITY',
        12: 'I-FACILITY',
        13: 'B-LOCATION-UNK',
        14: 'I-LOCATION-UNK',
        15: 'B-LOCATION-NAT',
        16: 'I-LOCATION-NAT'
    }


def convert_numeric_labels_to_iob(labels, label_mapping=None):
    """
    Μετατρέπει αριθμητικά labels σε IOB strings
    
    Args:
        labels: Λίστα με αριθμητικά labels
        label_mapping: Dictionary για μετατροπή αριθμών σε strings
    
    Returns:
        Λίστα με IOB string labels
    """
    if label_mapping is None:
        label_mapping = get_iob_label_mapping()
    
    converted_labels = []
    for label in labels:
        if isinstance(label, (int, float)):
            converted_labels.append(label_mapping.get(int(label), f'UNKNOWN_{int(label)}'))
        else:
            converted_labels.append(str(label))
    
    return converted_labels


def analyze_dataset(dataset_name, dataset_config=None, dataset_alias=None):
    """
    Αναλύει ένα συγκεκριμένο dataset
    
    Args:
        dataset_name (str): Όνομα του dataset
        dataset_config (str): Configuration του dataset (αν χρειάζεται)
        dataset_alias (str): Alias για το dataset στην αναφορά
    
    Returns:
        dict: Στατιστικά του dataset
    """
    
    alias = dataset_alias or dataset_name
    print(f"🔄 Φορτώνω το dataset: {alias}")
    
    try:
        # Φόρτωση dataset
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config)
        else:
            dataset = load_dataset(dataset_name)
        print(f"✅ Dataset {alias} φορτώθηκε επιτυχώς!")
        
    except Exception as e:
        print(f"❌ Σφάλμα κατά τη φόρτωση του dataset {alias}: {e}")
        return None
    
    # Αρχικοποίηση dictionary για αποτελέσματα
    results = {
        'dataset_info': {
            'name': dataset_name,
            'config': dataset_config,
            'alias': alias,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        'splits': {}
    }
    
    print(f"\n📊 Αναλύω τα splits του dataset {alias}...")
    
    # Ανάλυση κάθε split
    for split_name in dataset.keys():
        print(f"\n🔍 Αναλύω το split: {split_name}")
        
        split_data = dataset[split_name]
        num_documents = len(split_data)
        
        # Μετρητής για IOB tags
        iob_counter = Counter()
        total_tokens = 0
        
        # Ανάλυση κάθε document
        for i, example in enumerate(split_data):
            if i % 100 == 0 and i > 0:
                print(f"   Επεξεργασία document {i}/{num_documents}")
            
            # Εξαγωγή NER tags - προσαρμογή για διαφορετικά field names
            ner_tags = None
            if 'ner' in example:
                ner_tags = example['ner']
            elif 'ner_tags' in example:
                ner_tags = example['ner_tags']
            elif 'labels' in example:
                ner_tags = example['labels']
            elif 'label' in example:  # Για joelito/lextreme
                ner_tags = example['label']
            
            if ner_tags is None:
                if i < 10:  # Μόνο τα πρώτα 10 warnings
                    print(f"⚠️ Δεν βρέθηκαν NER tags στο example {i}")
                continue
                
            total_tokens += len(ner_tags)
            
            # Μετατροπή αριθμητικών labels σε IOB strings αν χρειάζεται
            if len(ner_tags) > 0 and isinstance(ner_tags[0], (int, float)):
                ner_tags = convert_numeric_labels_to_iob(ner_tags)
            
            # Μέτρηση κάθε tag
            for tag in ner_tags:
                iob_counter[tag] += 1
        
        # Αποθήκευση αποτελεσμάτων για το split
        results['splits'][split_name] = {
            'num_documents': num_documents,
            'total_tokens': total_tokens,
            'iob_distribution': dict(iob_counter),
            'unique_tags': len(iob_counter)
        }
        
        print(f"   ✅ {split_name}: {num_documents:,} documents, {total_tokens:,} tokens")
    
    return results


def generate_single_dataset_report(results):
    """
    Δημιουργεί αναλυτική αναφορά για ένα dataset
    
    Args:
        results (dict): Αποτελέσματα ανάλυσης
        
    Returns:
        str: Αναφορά σε μορφή κειμένου
    """
    
    if not results:
        return "❌ Δεν υπάρχουν αποτελέσματα για ανάλυση"
    
    report_lines = []
    
    # Header
    report_lines.append("="*80)
    report_lines.append(f"ΑΝΑΛΥΣΗ DATASET: {results['dataset_info']['alias'].upper()}")
    report_lines.append("="*80)
    report_lines.append(f"Dataset: {results['dataset_info']['name']}")
    if results['dataset_info']['config']:
        report_lines.append(f"Config: {results['dataset_info']['config']}")
    report_lines.append(f"Ημερομηνία ανάλυσης: {results['dataset_info']['analysis_date']}")
    report_lines.append("")
    
    # Συνολικά στατιστικά
    report_lines.append("📊 ΣΥΝΟΛΙΚΑ ΣΤΑΤΙΣΤΙΚΑ")
    report_lines.append("-" * 40)
    
    total_docs = sum(split_info['num_documents'] for split_info in results['splits'].values())
    total_tokens = sum(split_info['total_tokens'] for split_info in results['splits'].values())
    
    report_lines.append(f"Συνολικά documents: {total_docs:,}")
    report_lines.append(f"Συνολικά tokens: {total_tokens:,}")
    report_lines.append(f"Διαθέσιμα splits: {', '.join(results['splits'].keys())}")
    report_lines.append("")
    
    # Ανάλυση ανά split
    for split_name, split_info in results['splits'].items():
        report_lines.append(f"📁 SPLIT: {split_name.upper()}")
        report_lines.append("-" * 40)
        report_lines.append(f"Documents: {split_info['num_documents']:,}")
        report_lines.append(f"Tokens: {split_info['total_tokens']:,}")
        report_lines.append(f"Μοναδικά IOB tags: {split_info['unique_tags']}")
        
        # Ποσοστό του συνολικού dataset
        if total_docs > 0:
            doc_percentage = (split_info['num_documents'] / total_docs) * 100
        else:
            doc_percentage = 0
            
        if total_tokens > 0:
            token_percentage = (split_info['total_tokens'] / total_tokens) * 100
        else:
            token_percentage = 0
            
        report_lines.append(f"Ποσοστό documents: {doc_percentage:.2f}%")
        report_lines.append(f"Ποσοστό tokens: {token_percentage:.2f}%")
        report_lines.append("")
        
        # IOB Distribution
        report_lines.append("🏷️  ΚΑΤΑΝΟΜΗ IOB TAGS:")
        
        # Ταξινόμηση tags
        iob_dist = split_info['iob_distribution']
        sorted_tags = sorted(iob_dist.keys(), key=lambda x: (
            0 if x == 'O' else 1 if str(x).startswith('B-') else 2 if str(x).startswith('I-') else 3,
            str(x)
        ))
        
        for tag in sorted_tags:
            count = iob_dist[tag]
            if split_info['total_tokens'] > 0:
                percentage = (count / split_info['total_tokens']) * 100
            else:
                percentage = 0
            report_lines.append(f"  {str(tag):<20}: {count:>8,} ({percentage:>6.2f}%)")
        
        report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("ΤΕΛΟΣ ΑΝΑΦΟΡΑΣ")
    report_lines.append("="*80)
    
    return "\n".join(report_lines)


def generate_comparative_report(results1, results2):
    """
    Δημιουργεί συγκριτική αναφορά μεταξύ δύο datasets
    
    Args:
        results1 (dict): Αποτελέσματα πρώτου dataset
        results2 (dict): Αποτελέσματα δεύτερου dataset
        
    Returns:
        str: Συγκριτική αναφορά σε μορφή κειμένου
    """
    
    if not results1 or not results2:
        return "❌ Δεν υπάρχουν αποτελέσματα για σύγκριση"
    
    report_lines = []
    
    # Header
    report_lines.append("="*100)
    report_lines.append("ΣΥΓΚΡΙΤΙΚΗ ΑΝΑΛΥΣΗ GREEK LEGAL NER DATASETS")
    report_lines.append("="*100)
    report_lines.append(f"Dataset 1: {results1['dataset_info']['alias']} ({results1['dataset_info']['name']})")
    report_lines.append(f"Dataset 2: {results2['dataset_info']['alias']} ({results2['dataset_info']['name']})")
    report_lines.append(f"Ημερομηνία σύγκρισης: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Συνολικά στατιστικά σύγκρισης
    report_lines.append("📊 ΣΥΝΟΛΙΚΗ ΣΥΓΚΡΙΣΗ")
    report_lines.append("-" * 60)
    
    # Υπολογισμός συνολικών
    total_docs1 = sum(split_info['num_documents'] for split_info in results1['splits'].values())
    total_tokens1 = sum(split_info['total_tokens'] for split_info in results1['splits'].values())
    total_docs2 = sum(split_info['num_documents'] for split_info in results2['splits'].values())
    total_tokens2 = sum(split_info['total_tokens'] for split_info in results2['splits'].values())
    
    report_lines.append(f"{'Μετρική':<25} {'Dataset 1':<20} {'Dataset 2':<20} {'Διαφορά':<15}")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Συνολικά Docs':<25} {total_docs1:<20,} {total_docs2:<20,} {total_docs2-total_docs1:<+15,}")
    report_lines.append(f"{'Συνολικά Tokens':<25} {total_tokens1:<20,} {total_tokens2:<20,} {total_tokens2-total_tokens1:<+15,}")
    report_lines.append("")
    
    # Σύγκριση ανά split
    common_splits = set(results1['splits'].keys()) & set(results2['splits'].keys())
    
    if common_splits:
        report_lines.append("📁 ΣΥΓΚΡΙΣΗ ΑΝΑ SPLIT")
        report_lines.append("-" * 60)
        
        for split_name in sorted(common_splits):
            split1 = results1['splits'][split_name]
            split2 = results2['splits'][split_name]
            
            report_lines.append(f"\n🔍 SPLIT: {split_name.upper()}")
            report_lines.append("-" * 30)
            report_lines.append(f"{'Μετρική':<20} {'Dataset 1':<15} {'Dataset 2':<15} {'Διαφορά':<15}")
            report_lines.append("-" * 65)
            
            docs_diff = split2['num_documents'] - split1['num_documents']
            tokens_diff = split2['total_tokens'] - split1['total_tokens']
            tags_diff = split2['unique_tags'] - split1['unique_tags']
            
            report_lines.append(f"{'Documents':<20} {split1['num_documents']:<15,} {split2['num_documents']:<15,} {docs_diff:<+15,}")
            report_lines.append(f"{'Tokens':<20} {split1['total_tokens']:<15,} {split2['total_tokens']:<15,} {tokens_diff:<+15,}")
            report_lines.append(f"{'Unique Tags':<20} {split1['unique_tags']:<15} {split2['unique_tags']:<15} {tags_diff:<+15}")
    
    # Σύγκριση IOB tags
    report_lines.append("\n\n🏷️  ΣΥΓΚΡΙΣΗ IOB DISTRIBUTIONS")
    report_lines.append("-" * 80)
    
    # Συλλογή όλων των tags από τα δύο datasets
    all_tags1 = set()
    all_tags2 = set()
    
    for split_info in results1['splits'].values():
        all_tags1.update(split_info['iob_distribution'].keys())
    
    for split_info in results2['splits'].values():
        all_tags2.update(split_info['iob_distribution'].keys())
    
    all_tags = all_tags1 | all_tags2
    
    # Ταξινόμηση tags
    sorted_all_tags = sorted(all_tags, key=lambda x: (
        0 if str(x) == 'O' else 1 if str(x).startswith('B-') else 2 if str(x).startswith('I-') else 3,
        str(x)
    ))
    
    report_lines.append(f"{'Tag':<20} {'Dataset 1':<15} {'Dataset 2':<15} {'Κοινό':<10}")
    report_lines.append("-" * 60)
    
    for tag in sorted_all_tags:
        in_ds1 = "✓" if tag in all_tags1 else "✗"
        in_ds2 = "✓" if tag in all_tags2 else "✗"
        common = "✓" if tag in all_tags1 and tag in all_tags2 else "✗"
        
        report_lines.append(f"{str(tag):<20} {in_ds1:<15} {in_ds2:<15} {common:<10}")
    
    # Στατιστικά tags
    common_tags = all_tags1 & all_tags2
    only_ds1 = all_tags1 - all_tags2
    only_ds2 = all_tags2 - all_tags1
    
    report_lines.append("\n📈 ΣΤΑΤΙΣΤΙΚΑ TAGS")
    report_lines.append("-" * 30)
    report_lines.append(f"Κοινά tags: {len(common_tags)}")
    report_lines.append(f"Μόνο στο Dataset 1: {len(only_ds1)}")
    report_lines.append(f"Μόνο στο Dataset 2: {len(only_ds2)}")
    
    if only_ds1:
        report_lines.append(f"Tags μόνο στο Dataset 1: {', '.join(map(str, sorted(only_ds1)))}")
    
    if only_ds2:
        report_lines.append(f"Tags μόνο στο Dataset 2: {', '.join(map(str, sorted(only_ds2)))}")
    
    report_lines.append("")
    report_lines.append("="*100)
    report_lines.append("ΤΕΛΟΣ ΣΥΓΚΡΙΤΙΚΗΣ ΑΝΑΦΟΡΑΣ")
    report_lines.append("="*100)
    
    return "\n".join(report_lines)


def save_report_to_file(report, output_file):
    """
    Αποθηκεύει αναφορά σε txt αρχείο
    
    Args:
        report (str): Αναφορά σε μορφή κειμένου
        output_file (str): Όνομα αρχείου εξόδου
    """
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Αναφορά αποθηκεύτηκε στο αρχείο: {output_file}")
        
    except Exception as e:
        print(f"❌ Σφάλμα κατά την αποθήκευση: {e}")


def main():
    """
    Κύρια συνάρτηση εκτέλεσης
    """
    
    print("🚀 Εκκίνηση συγκριτικής ανάλυσης Greek Legal NER Datasets")
    print("=" * 80)
    
    # Αναλύσεις datasets
    print("\n1️⃣ Ανάλυση πρώτου dataset...")
    results1 = analyze_dataset(
        "joelniklaus/greek_legal_ner", 
        dataset_alias="joelniklaus/greek_legal_ner"
    )
    
    print("\n2️⃣ Ανάλυση δεύτερου dataset...")
    results2 = analyze_dataset(
        "joelito/lextreme", 
        "greek_legal_ner",
        dataset_alias="joelito/lextreme"
    )
    
    if not results1 or not results2:
        print("❌ Δεν μπόρεσα να φορτώσω όλα τα datasets")
        return
    
    # Δημιουργία timestamp για αρχεία
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Δημιουργία αναφορών
    print("\n📋 Δημιουργία αναφορών...")
    
    # 1. Αναφορά για το δεύτερο dataset
    report2 = generate_single_dataset_report(results2)
    output_file2 = f"lextreme_greek_legal_ner_analysis_{timestamp}.txt"
    save_report_to_file(report2, output_file2)
    
    # 2. Συγκριτική αναφορά
    comparative_report = generate_comparative_report(results1, results2)
    comparative_file = f"datasets_comparison_{timestamp}.txt"
    save_report_to_file(comparative_report, comparative_file)
    
    # Εμφάνιση σύντομης περίληψης
    print("\n📊 ΣΥΝΤΟΜΗ ΠΕΡΙΛΗΨΗ ΣΥΓΚΡΙΣΗΣ:")
    print("-" * 50)
    
    # Dataset 1
    total_docs1 = sum(split_info['num_documents'] for split_info in results1['splits'].values())
    total_tokens1 = sum(split_info['total_tokens'] for split_info in results1['splits'].values())
    
    # Dataset 2  
    total_docs2 = sum(split_info['num_documents'] for split_info in results2['splits'].values())
    total_tokens2 = sum(split_info['total_tokens'] for split_info in results2['splits'].values())
    
    print(f"Dataset 1 (joelniklaus): {total_docs1:,} docs, {total_tokens1:,} tokens")
    print(f"Dataset 2 (joelito):     {total_docs2:,} docs, {total_tokens2:,} tokens")
    print(f"Διαφορά:                {total_docs2-total_docs1:+,} docs, {total_tokens2-total_tokens1:+,} tokens")
    
    print(f"\n💾 Αρχεία που δημιουργήθηκαν:")
    print(f"  - {output_file2}")
    print(f"  - {comparative_file}")


if __name__ == "__main__":
    main()
