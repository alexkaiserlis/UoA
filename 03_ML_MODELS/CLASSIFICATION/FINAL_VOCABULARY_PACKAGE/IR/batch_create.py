#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Summary Creator - Δημ    print(f"\n📊 ΣΤΑΤΙΣΤΙΚΑ:")
    print(f"   • Concepts προς επεξεργασία: {len(concept_ids):,}")
    print(f"   • Εκτιμώμενος χρόνος: {len(concept_ids) * 3.3 / 60:.1f} λεπτά")
    
    # Ακριβέστερος υπολογισμός κόστους
    # gpt-4o-mini: $0.150/1M input tokens, $0.600/1M output tokens
    avg_input_tokens = 1600   # Βασισμένο στα logs
    avg_output_tokens = 150   # ~100 words = ~150 tokens
    
    total_input_tokens = len(concept_ids) * avg_input_tokens
    total_output_tokens = len(concept_ids) * avg_output_tokens
    
    input_cost = (total_input_tokens / 1_000_000) * 0.150
    output_cost = (total_output_tokens / 1_000_000) * 0.600
    total_cost = input_cost + output_cost
    
    print(f"   • Εκτιμώμενο κόστος OpenAI:")
    print(f"     - Input tokens: {total_input_tokens:,} × $0.150/1M = ${input_cost:.3f}")
    print(f"     - Output tokens: {total_output_tokens:,} × $0.600/1M = ${output_cost:.3f}")
    print(f"     - 💰 ΣΥΝΟΛΟ: ${total_cost:.2f}")γία πολλαπλών περιλήψεων
"""

import os
import time
import json
from datetime import datetime
from eurovoc_summary_generator import EurovocSummaryGenerator

# Pricing για διαφορετικά μοντέλα ($ per 1M tokens)
MODEL_PRICING = {
    'gpt-4o-mini': {
        'input': 0.150,
        'output': 0.600
    },
    'gpt-3.5-turbo': {
        'input': 0.500,
        'output': 1.500
    },
    'gpt-4-turbo': {
        'input': 10.000,
        'output': 30.000
    },
    'gpt-4': {
        'input': 30.000,
        'output': 60.000
    }
}

def estimate_tokens(text, is_output=False):
    """Εκτιμά τον αριθμό tokens σε ένα κείμενο."""
    # Απλή εκτίμηση: ~4 χαρακτήρες = 1 token
    # Για output είναι πιο συμπαγές
    chars_per_token = 3.5 if is_output else 4
    return int(len(text) / chars_per_token)

def calculate_request_cost(prompt, response, model):
    """Υπολογίζει το κόστος μιας συγκεκριμένης request."""
    if model not in MODEL_PRICING:
        return 0.0
    
    input_tokens = estimate_tokens(prompt, is_output=False)
    output_tokens = estimate_tokens(response, is_output=True)
    
    pricing = MODEL_PRICING[model]
    input_cost = (input_tokens / 1_000_000) * pricing['input']
    output_cost = (output_tokens / 1_000_000) * pricing['output']
    
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': input_cost + output_cost
    }

def save_cost_report(run_data, cost_summary):
    """Αποθηκεύει αναφορά κόστους."""
    cost_file = os.path.join(os.path.dirname(__file__), 'cost_tracking.json')
    
    # Φόρτωση υπάρχοντων δεδομένων
    if os.path.exists(cost_file):
        with open(cost_file, 'r', encoding='utf-8') as f:
            cost_data = json.load(f)
    else:
        cost_data = {
            'total_runs': 0,
            'total_cost': 0.0,
            'total_concepts': 0,
            'total_tokens': {'input': 0, 'output': 0},
            'runs': []
        }
    
    # Προσθήκη νέου run
    cost_data['runs'].append(run_data)
    cost_data['total_runs'] += 1
    cost_data['total_cost'] += cost_summary['total_cost']
    cost_data['total_concepts'] += cost_summary['concepts_processed']
    cost_data['total_tokens']['input'] += cost_summary['total_input_tokens']
    cost_data['total_tokens']['output'] += cost_summary['total_output_tokens']
    
    # Αποθήκευση
    with open(cost_file, 'w', encoding='utf-8') as f:
        json.dump(cost_data, f, ensure_ascii=False, indent=2)
    
    return cost_file

def create_multiple_summaries():
    """Δημιουργεί περιλήψεις για πολλαπλά concepts ή όλα τα concepts."""
    
    print("📚 BATCH SUMMARY CREATOR")
    print("="*50)
    
    # Δημιουργία generator
    generator = EurovocSummaryGenerator()
    
    # Φόρτωση δεδομένων
    print("📥 Φόρτωση δεδομένων...")
    generator.load_data()
    
    print(f"\n📊 ΣΤΑΤΙΣΤΙΚΑ VOCABULARY:")
    print(f"   • Συνολικές λέξεις: {len(generator.vocabulary):,}")
    print(f"   • Concepts διαθέσιμα: {len(generator.enhanced_mapping):,}")
    
    # Επιλογές δημιουργίας
    print("\n🎯 ΕΠΙΛΟΓΕΣ ΔΗΜΙΟΥΡΓΙΑΣ:")
    print("1. Specific concepts (sample list)")
    print("2. ΟΛΑ τα Eurovoc concepts (7,384 concepts)")
    print("3. First 100 concepts (για δοκιμή)")
    print("4. Custom word count per concept")
    print("4. Concepts από συγκεκριμένη κατηγορία")
    
    # Production ready - μπορείς να αλλάξεις εδώ
    choice = "2"  # Αλλαξε σε "1", "2", "3" ή "4"
    
    if choice == "1":
        # Συγκεκριμένα concepts για cost tracking test
        concept_ids = [
            "1033",  # test concept 1
            "1034",  # test concept 2
        ]
    elif choice == "2":
        # ΟΛΑ τα concepts
        concept_ids = list(generator.enhanced_mapping.keys())
        print(f"🚀 Θα δημιουργήσουμε περιλήψεις για ΟΛΑ τα {len(concept_ids):,} concepts!")
        print("⚠️  Αυτό θα πάρει πολύ χρόνο και κόστος OpenAI!")
        
    elif choice == "3":
        # Πρώτα 100 για δοκιμή
        concept_ids = list(generator.enhanced_mapping.keys())[:100]
        print(f"🧪 ΔΟΚΙΜΗ: Θα δημιουργήσουμε περιλήψεις για τα πρώτα {len(concept_ids)} concepts")
        
    elif choice == "4":
        # Custom word count per concept
        max_words = input("Αριθμός λέξεων ανά concept (default: 150): ").strip()
        max_words = int(max_words) if max_words.isdigit() else 150
        
        concept_ids = [
            "5132",  # health control
            "1030",  # plant life
            "192",   # agricultural statistics
            "4041",  # energy policy
            "6066",  # social security
        ]
        print(f"🎯 Custom test με {max_words} λέξεις ανά concept")
        
    else:
        print("❌ Μη έγκυρη επιλογή!")
        return
    
    # Ρύθμιση παραμέτρων
    max_vocabulary_words = 150  # Default
    if choice == "4":
        max_vocabulary_words = max_words if 'max_words' in locals() else 150
    
    print(f"\n� ΣΤΑΤΙΣΤΙΚΑ:")
    print(f"   • Concepts προς επεξεργασία: {len(concept_ids):,}")
    print(f"   • Εκτιμώμενος χρόνος: {len(concept_ids) * 3 / 60:.1f} λεπτά")
    print(f"   • Εκτιμώμενο κόστος OpenAI: ~${len(concept_ids) * 0.002:.2f}")
    
    # Επιβεβαίωση
    print(f"\n⚠️  ΠΡΟΣΟΧΗ: Αυτό θα δημιουργήσει {len(concept_ids)} αρχεία JSON!")
    confirmation = input("Συνέχεια; (y/N): ").lower().strip()
    
    if confirmation != 'y':
        print("❌ Ακυρώθηκε από τον χρήστη")
        return
    
    # Batch επεξεργασία με advanced tracking
    print(f"\n🚀 Εκκίνηση batch επεξεργασίας...")
    print(f"📊 Progress tracking ενεργοποιημένο")
    print(f"💰 Cost tracking ενεργοποιημένο")
    
    # Custom batch function με καλύτερο tracking
    successful = 0
    failed = 0
    skipped = 0
    start_time = time.time()
    
    # Cost tracking
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    cost_details = []
    
    for i, concept_id in enumerate(concept_ids, 1):
        try:
            # Progress update κάθε 10 concepts
            if i % 10 == 0 or i == 1:
                elapsed = time.time() - start_time
                rate = i / elapsed * 60 if elapsed > 0 else 0
                eta = (len(concept_ids) - i) / (rate / 60) if rate > 0 else 0
                print(f"📈 Progress: {i:,}/{len(concept_ids):,} ({i/len(concept_ids)*100:.1f}%) | "
                      f"Rate: {rate:.1f}/min | ETA: {eta/60:.1f}h")
            
            # Έλεγχος αν υπάρχει ήδη το αρχείο
            output_file = os.path.join(generator.output_dir, f"concept_{concept_id}_summary.json")
            if os.path.exists(output_file):
                print(f"⏭️  Skipping {concept_id} (already exists)")
                skipped += 1
                continue
            
            # Δημιουργία περίληψης
            concept_data = generator.enhanced_mapping[concept_id]
            related_words, word_stats = generator.get_related_vocabulary_words(concept_id, max_vocabulary_words)
            prompt = generator.create_llm_prompt(concept_data, related_words, word_stats, 'medium')
            
            result = generator.create_summary_for_concept(
                concept_id=concept_id,
                summary_length='medium',
                model='gpt-4o-mini',
                max_words=max_vocabulary_words
            )
            
            # Υπολογισμός κόστους για αυτή την request
            request_cost = calculate_request_cost(prompt, result['summary'], 'gpt-4o-mini')
            total_cost += request_cost['total_cost']
            total_input_tokens += request_cost['input_tokens']
            total_output_tokens += request_cost['output_tokens']
            
            cost_details.append({
                'concept_id': concept_id,
                'concept_title': concept_data['title'],
                'cost': request_cost['total_cost'],
                'tokens': {
                    'input': request_cost['input_tokens'],
                    'output': request_cost['output_tokens']
                }
            })
            
            # Αποθήκευση
            generator.save_summary(result)
            successful += 1
            
            # Rate limiting
            time.sleep(2.0)
            
        except KeyboardInterrupt:
            print(f"\n🛑 Διακοπή από χρήστη στο concept {i}/{len(concept_ids)}")
            break
        except Exception as e:
            print(f"❌ Error processing {concept_id}: {str(e)}")
            failed += 1
            continue
    
    # Τελικά στατιστικά με cost tracking
    total_time = time.time() - start_time
    
    # Δημιουργία cost summary
    cost_summary = {
        'total_cost': total_cost,
        'concepts_processed': successful,
        'concepts_skipped': skipped,
        'concepts_failed': failed,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'cost_per_concept': total_cost / successful if successful > 0 else 0
    }
    
    # Δεδομένα run
    run_data = {
        'timestamp': datetime.now().isoformat(),
        'model_used': 'gpt-4o-mini',
        'duration_minutes': total_time / 60,
        'concepts_requested': len(concept_ids),
        'concepts_processed': successful,
        'concepts_skipped': skipped,
        'concepts_failed': failed,
        'cost_summary': cost_summary,
        'cost_details': cost_details[:10]  # Αποθήκευση μόνο των πρώτων 10 για μέγεθος
    }
    
    print(f"\n✅ BATCH ΟΛΟΚΛΗΡΩΘΗΚΕ!")
    print(f"📊 ΣΤΑΤΙΣΤΙΚΑ:")
    print(f"   • Επιτυχής: {successful:,}")
    print(f"   • Παραλήφθηκαν: {skipped:,}")
    print(f"   • Αποτυχία: {failed:,}")
    print(f"   • Συνολικός χρόνος: {total_time/60:.1f} λεπτά")
    print(f"   • Μέσος όρος: {total_time/max(successful+failed,1):.1f} δευτ/concept")
    
    print(f"\n💰 ΚΟΣΤΟΣ ΑΝΑΛΥΣΗ:")
    print(f"   • Συνολικό κόστος: ${total_cost:.4f}")
    print(f"   • Κόστος ανά concept: ${cost_summary['cost_per_concept']:.6f}")
    print(f"   • Input tokens: {total_input_tokens:,}")
    print(f"   • Output tokens: {total_output_tokens:,}")
    print(f"   • Total tokens: {total_input_tokens + total_output_tokens:,}")
    
    # Αποθήκευση cost report
    if successful > 0:
        cost_file = save_cost_report(run_data, cost_summary)
        print(f"   • 📄 Cost report αποθηκεύτηκε: {cost_file}")
    
    print(f"\n📁 Αρχεία αποθηκεύτηκαν στο: {generator.output_dir}")

if __name__ == "__main__":
    create_multiple_summaries()
