from datasets import load_dataset
import json
import os

# Φόρτωση του dataset
dataset = load_dataset("joelito/lextreme", "greek_legal_ner", trust_remote_code=True)

# Ο φάκελος που τρέχει το script
output_dir = os.path.dirname(os.path.abspath(__file__))

# Εκτύπωση πληροφοριών για το dataset
print("Dataset splits:", list(dataset.keys()))
print("Dataset features:", dataset['train'].features if 'train' in dataset else "No train split")

# Αποθήκευση κάθε split σε JSON
for split_name, split_data in dataset.items():
    output_path = os.path.join(output_dir, f"{split_name}.json")
    
    # Μετατροπή σε dictionary για αποθήκευση
    split_dict = split_data.to_dict()
    
    print(f"\nSplit '{split_name}':")
    print(f"  - Αριθμός εγγραφών: {len(split_dict['tokens'])}")
    print(f"  - Πεδία: {list(split_dict.keys())}")
    
    # Έλεγχος αν υπάρχουν labels
    if 'ner_tags' in split_dict:
        print(f"  - Unique NER tags: {set([tag for tags in split_dict['ner_tags'] for tag in tags])}")
    elif 'label' in split_dict:
        print(f"  - Label field exists: {type(split_dict['label'])}")
    
    # Αποθήκευση
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, ensure_ascii=False, indent=2)
    
    print(f"  - Αποθηκεύτηκε στο: {output_path}")

print("\nΤέλος επεξεργασίας!")