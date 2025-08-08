# ARCHIVE UTILITIES 🗂️

**Φάκελος αρχειοθέτησης εργαλείων ανάπτυξης και ανάλυσης**

Αυτός ο φάκελος περιέχει scripts και εργαλεία που χρησιμοποιήθηκαν κατά την ανάπτυξη του Eurovision Legal Vocabulary Package, αλλά δεν χρειάζονται στην καθημερινή χρήση του συστήματος.

## 📁 Δομή

```
ARCHIVE_UTILITIES/
├── 📂 development_scripts/     # Scripts που χρησιμοποιήθηκαν για την κατασκευή του package
├── 📂 analysis_tools/          # Εργαλεία ανάλυσης και διερεύνησης δεδομένων
├── 📂 logs/                   # Log αρχεία από την ανάπτυξη
└── 📄 README.md               # Αυτό το αρχείο
```

## 🔧 Development Scripts

**Περιεχόμενα:** Scripts που δημιούργησαν το enhanced mapping και άλλα core components

- `create_enhanced_mapping.py` - Δημιουργία εμπλουτισμένου Eurovision mapping
- `extract_eurovoc_id_title_mappings.py` - Εξαγωγή ID→Title mappings από CSV
- `generate_statistics.py` - Υπολογισμός στατιστικών του vocabulary
- `package_summary.py` - Δημιουργία package summary
- `final_summary.py` - Τελική αναφορά package

**Χρήση:** Αυτά τα scripts εκτελέστηκαν κατά την κατασκευή του package και δεν χρειάζονται πλέον για κανονική λειτουργία.

## 📊 Analysis Tools

**Περιεχόμενα:** Εργαλεία για ανάλυση και εξερεύνηση των δεδομένων

- `analyze_concept.py` - Ανάλυση συγκεκριμένων Eurovision concepts
- `analyze_eurovoc_csv.py` - Ανάλυση του αρχικού Eurovision CSV
- `compare_concepts.py` - Σύγκριση μεταξύ concepts
- `explain_eurovision_concepts.py` - Εξήγηση Eurovision structure
- `explain_mapping_structure.py` - Λεπτομερής εξήγηση mapping structure

**Χρήση:** Χρήσιμα για research και διερεύνηση των δεδομένων, αλλά όχι για production.

## 📄 Logs

**Περιεχόμενα:** Log αρχεία από την ανάπτυξη

- `eurovoc_id_title_extraction.log` - Logs από το extraction process

## ⚠️ Σημαντικό

Τα αρχεία σε αυτόν τον φάκελο:
- ✅ Μπορούν να διαγραφούν χωρίς να επηρεάσουν τη λειτουργία του συστήματος
- ✅ Κρατιούνται για ιστορικούς λόγους και αναφορά
- ✅ Μπορούν να χρησιμοποιηθούν για debugging/research

## 🚀 Κύρια Συστήματα (Εκτός Archive)

Για κανονική χρήση του package, χρησιμοποιήστε:

- **`data/`** - Τα κύρια δεδομένα (vocabulary, mappings)
- **`IR/`** - Information Retrieval σύστημα με LLM
- **`scripts/`** - Κενός (όλα μεταφέρθηκαν εδώ)
- **`README.md`** - Κύρια τεκμηρίωση

---

**📅 Αρχειοθετήθηκε:** 29 Ιουλίου 2025  
**🎯 Σκοπός:** Clean organization του final package
