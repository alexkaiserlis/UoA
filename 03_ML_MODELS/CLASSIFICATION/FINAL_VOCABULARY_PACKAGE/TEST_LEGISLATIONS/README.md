# TEST_LEGISLATIONS Folder

## 📋 Σκοπός
Αυτός ο φάκελος προορίζεται για PDF αρχεία νομοθετικών κειμένων που θέλετε να δοκιμάσετε με τον Eurovoc TF-IDF classifier.

## 📁 Τι να προσθέσετε εδώ
- **PDF αρχεία** νομοθετικών κειμένων (νόμοι, κανονισμοί, οδηγίες)
- **Προτιμητέα γλώσσα:** Αγγλικά (αλλά δουλεύει και με ελληνικά)
- **Μέγεθος:** Οποιοδήποτε (το σύστημα παίρνει τις πρώτες ~30,000 χαρακτήρες)

## 📄 Παραδείγματα αρχείων
```
TEST_LEGISLATIONS/
├── EU_Regulation_2021_123.pdf      # EU κανονισμός
├── Greek_Law_4820_2021.pdf         # Ελληνικός νόμος  
├── Directive_2020_456.pdf          # EU οδηγία
└── Trade_Agreement_2022.pdf        # Εμπορική συμφωνία
```

## 🚀 Πως να τα χρησιμοποιήσετε

### Μέθοδος 1: Απλή δοκιμή
```bash
cd IR/
python eurovoc_classifier.py
```

### Μέθοδος 2: Εξειδικευμένη ανάλυση (Προτεινόμενο)
```bash
cd IR/
python test_pdf_classifier.py
```

## 🎯 Τι θα δείτε
- Αυτόματη εξαγωγή κειμένου από PDF
- Classification σε Eurovoc concepts
- Top-10/15 πιο σχετικά concepts με scores
- Interactive evaluation αν ξέρετε τα σωστά concepts

## 📦 Prerequisites
Βεβαιωθείτε ότι έχετε εγκαταστήσει:
```bash
pip install PyPDF2 PyMuPDF
```

## 💡 Tips
1. **Καλύτερα αποτελέσματα:** Χρησιμοποιήστε PDF με καθαρό κείμενο (όχι scanned images)
2. **Eurovoc concepts:** Αν ξέρετε τα σωστά concepts, εισάγετέ τα για evaluation
3. **Μέγεθος:** Για καλύτερη απόδοση, χρησιμοποιήστε 3-5 αρχεία κάθε φορά

## 🔍 Evaluation
Όταν εισάγετε γνωστά Eurovoc concepts, το σύστημα θα υπολογίσει:
- Recall@1, @3, @5, @10
- Matches στα top results  
- Confidence analysis

---
**Σημείωση:** Αυτός ο φάκελος δημιουργήθηκε αυτόματα από το IR system. Προσθέστε PDF αρχεία και τρέξτε το testing!
