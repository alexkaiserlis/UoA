# 🚀 Multi-Run Experiment Feature

## 📋 Περιγραφή

Αυτό το feature επιτρέπει να τρέχεις το ίδιο experiment πολλές φορές με διαφορετικά seeds και να αθροίζει τα αποτελέσματα για στατιστική αξιοπιστία.

## 🎯 Τι κάνει

- **Πολλαπλά runs**: Τρέχει το ίδιο experiment 6 φορές (ή όσες θέλεις)
- **Seed management**: Ξεκινάει από το seed που θες και το αυξάνει κατά 1 κάθε φορά
- **Αποθήκευση αποτελεσμάτων**: Κάθε run αποθηκεύεται σε χωριστό φάκελο
- **Αθροιστικά στατιστικά**: Υπολογίζει mean, std, min, max για όλα τα F1 scores
- **Error handling**: Αν αποτύχει ένα run, συνεχίζει με τα επόμενα

## 📊 Τι αποτελέσματα παίρνεις

### Για κάθε run (π.χ. run_1_seed42/):
```
experiments/multirun/
├── run_1_seed42/
│   ├── model/              # Το εκπαιδευμένο μοντέλο
│   ├── results.json        # Αποτελέσματα αυτού του run
│   └── logs/              # Logs αυτού του run
├── run_2_seed43/
│   └── ...
└── aggregated_results.json  # ΚΥΡΙΟ ΑΡΧΕΙΟ με στατιστικά
```

### Στο aggregated_results.json:
```json
{
  "aggregated_results": {
    "overall_f1": {
      "mean": 0.8567,
      "std": 0.0123,
      "min": 0.8431,
      "max": 0.8698,
      "values": [0.8567, 0.8543, 0.8598, 0.8431, 0.8698, 0.8565]
    },
    "entity_metrics": {
      "PERSON_f1": {
        "mean": 0.9234,
        "std": 0.0087,
        ...
      },
      "ORGANIZATION_f1": { ... },
      "LOCATION_f1": { ... }
    }
  }
}
```

## 🛠️ Πώς να το χρησιμοποιήσεις

### Μέθοδος 1: Command Line Arguments

```bash
# 6 runs με seeds 42, 43, 44, 45, 46, 47
python main_script.py --preset CRF_ENHANCED --num-runs 6 --base-seed 42

# 5 runs με seeds 100, 110, 120, 130, 140 (seed-increment=10)
python main_script.py --preset CRF_ENHANCED --num-runs 5 --base-seed 100 --seed-increment 10

# Single run (default behavior)
python main_script.py --preset CRF_ENHANCED
```

### Μέθοδος 2: Configuration File

Το test δημιούργησε ένα `sample_multirun_config.json` αρχείο:

```bash
python main_script.py --config sample_multirun_config.json
```

## ⚙️ Configuration Parameters

```python
# Στο config.py ή στο JSON αρχείο:
{
  "num_runs": 6,                    # Πόσα runs να κάνει
  "base_seed": 42,                  # Αρχικό seed
  "seed_increment": 1,              # Πόσο να αυξάνει το seed κάθε φορά
  "aggregate_results": true,        # Να υπολογίσει στατιστικά
  "save_individual_models": true,   # Να αποθηκεύσει κάθε μοντέλο ξεχωριστά
  "continue_on_failure": false      # Να συνεχίσει αν αποτύχει ένα run
}
```

## 📈 Παράδειγμα αποτελεσμάτων

Μετά από 6 runs, θα δεις:

```
✅ MULTI-RUN EXPERIMENT COMPLETED
   Duration: 2:34:12
   Successful runs: 6/6
   Overall F1: 0.8567 ± 0.0123
```

## 🔧 Troubleshooting

### Αν αποτύχει ένα run:
- Το σύστημα θα συνεχίσει με τα επόμενα (αν `continue_on_failure=true`)
- Θα αποθηκεύσει partial results για το failed run
- Θα υπολογίσει στατιστικά μόνο από τα επιτυχημένα runs

### Για γρήγορο testing:
```python
# Στο config, βάλε μικρές τιμές:
config.training.max_steps = 100        # Λίγα steps
config.training.num_train_epochs = 1   # 1 epoch μόνο
config.num_runs = 3                    # 3 runs αντί για 6
```

## 🎯 Συνιστώμενη χρήση

Για τα πραγματικά experiments σου:

```bash
# 6 runs για στατιστική αξιοπιστία
python main_script.py --preset CRF_ENHANCED --num-runs 6 --base-seed 42 --output-dir ./final_experiments
```

## 📝 Σημειώσεις

- Κάθε run παίρνει περίπου το ίδιο χρόνο με ένα κανονικό experiment
- 6 runs = 6x το χρόνο, αλλά πολύ πιο αξιόπιστα αποτελέσματα
- Τα aggregated statistics είναι αυτά που θα χρησιμοποιήσεις στα papers σου
- Individual runs μπορείς να τα δεις για debugging ή analysis

**Το πιο σημαντικό αρχείο είναι το `aggregated_results.json` με τα στατιστικά!** 📊
