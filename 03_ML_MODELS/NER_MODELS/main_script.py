"""
Greek Legal NER - Main Training Script

This is the main training script for Greek Legal Named Entity Recognition.
It uses a modular architecture with separate components for model, data processing,
configuration, and evaluation.

🚀 MODULAR ARCHITECTURE:
=======================
- config.py: Centralized configuration management
- crf_model.py: ROBERTa + CRF model implementation
- focal_loss.py: Advanced loss functions for imbalanced data
- flexible_trainer.py: Enhanced trainer with comprehensive metrics
- data_utils.py: Data processing and augmentation utilities
- evaluation_metrics.py: Advanced evaluation and analysis

🎯 EXPERIMENT WORKFLOW:
======================
1. Load configuration (preset or custom)
2. Load and validate dataset
3. Process data (tokenization, augmentation, class weights)
4. Initialize model and trainer
5. Train with comprehensive monitoring
6. Evaluate with detailed analysis
7. Save results and model

📊 USAGE EXAMPLES:
=================
```bash
# Run baseline experiment
python main_script.py --preset BASELINE

# Run CRF enhanced experiment  
python main_script.py --preset CRF_ENHANCED

# Run with custom configuration
python main_script.py --config custom_config.json

# Run multiple experiments
python main_script.py --preset BASELINE CRF_ENHANCED WEIGHTED
```
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
import json
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    DataCollatorForTokenClassification,
    TrainingArguments,
    set_seed
)

# Import our custom modules
from config import ExperimentConfig, list_available_presets, get_config_summary
from crf_model import BertCRFForTokenClassification, create_crf_model
from focal_loss import create_focal_loss
from flexible_trainer import FlexibleTrainer
from data_utils import (
    tokenize_and_align_labels,
    compute_class_weights,
    apply_rare_class_augmentation,
    validate_dataset_structure,
    print_dataset_summary
)
from evaluation_metrics import (
    compute_metrics_with_model_info,
    export_detailed_results,
    NERMetricsCalculator
)


# =============================================================================
# JSON SERIALIZATION HELPER
# =============================================================================

def clean_for_json_serialization(obj):
    """
    Recursively clean data structure to make it JSON serializable.
    Converts NumPy types to native Python types.
    
    Args:
        obj: Object to clean
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {key: clean_for_json_serialization(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json_serialization(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    else:
        return obj


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup file handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger(__name__)


# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

class GreekLegalNERDataProcessor:
    """
    Data processor for Greek Legal NER dataset.
    
    Handles dataset loading, validation, tokenization, and preparation
    for training with comprehensive error checking and statistics.
    """
    
    def __init__(self, config: ExperimentConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.tokenizer = None
        self.dataset = None
        self.tokenized_dataset = None
        self.label_list = None
        self.id2label = None
        self.label2id = None
        self.class_weights = None
    
    def load_and_process_data(self):
        """
        Complete data loading and processing pipeline.
        
        🔍 PROCESSING PIPELINE:
        ======================
        1. Load raw dataset from HuggingFace
        2. Validate dataset structure and labels
        3. Initialize tokenizer
        4. Process labels and create mappings
        5. Tokenize and align labels with sliding window
        6. Apply data augmentation (if enabled)
        7. Compute class weights (if enabled)
        8. Print comprehensive statistics
        
        Returns:
            Tuple of (tokenized_dataset, tokenizer, label_mappings, class_weights)
        """
        
        self.logger.info("🚀 Starting data loading and processing pipeline")
        
        # 1. Load dataset
        self._load_dataset()
        
        # 2. Validate dataset structure
        self._validate_dataset()
        
        # 3. Initialize tokenizer
        self._initialize_tokenizer()
        
        # 4. Process labels
        self._process_labels()
        
        # 5. Tokenize data
        self._tokenize_data()
        
        # 6. Apply augmentation
        if self.config.data.enable_augmentation:
            self._apply_augmentation()
        
        # 7. Compute class weights
        if self.config.data.compute_class_weights:
            self._compute_class_weights()
        
        # 8. Print final statistics
        self._print_final_statistics()
        
        self.logger.info("✅ Data processing pipeline completed successfully")
        
        return (
            self.tokenized_dataset, 
            self.tokenizer, 
            {'id2label': self.id2label, 'label2id': self.label2id, 'label_list': self.label_list},
            self.class_weights
        )
    
    def _load_dataset(self):
        """Load the Greek Legal NER dataset."""
        self.logger.info(f"📚 Loading dataset: {self.config.data.dataset_name}")
        
        try:
            # Check if we should use local IOB files
            if self.config.data.dataset_name == "local_iob":
                self.logger.info("🔄 Loading local IOB files")
                self.dataset = self._load_local_iob_dataset()
            else:
                # Use HuggingFace dataset
                self.dataset = load_dataset(
                    self.config.data.dataset_name,
                    config=self.config.data.dataset_config,
                    cache_dir=self.config.data.cache_dir
                )
            
            self.logger.info(f"✅ Dataset loaded successfully")
            self.logger.info(f"   Train examples: {len(self.dataset['train']):,}")
            self.logger.info(f"   Test examples: {len(self.dataset['test']):,}")
            if 'validation' in self.dataset:
                self.logger.info(f"   Validation examples: {len(self.dataset['validation']):,}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load dataset: {e}")
            raise
    
    def _load_local_iob_dataset(self):
        """Load local IOB files and convert to HuggingFace dataset format."""
        # Base path for IOB files (absolute path)
        base_path = Path(r"c:\Users\User\Το Drive μου\AEGEAN UNIVERSITY\LEGAL DOCUMENTS ARCHIVE\ΠΑΙΓΑΙΟΥ\CODE\01_DATASETS\GREEK_LEGAL_NER")
        
        files = {
            "train": base_path / "train_iob.json",
            "test": base_path / "test_combined_iob.json", 
            "validation": base_path / "validation_iob.json"
        }
        
        datasets = {}
        
        for split_name, file_path in files.items():
            self.logger.info(f"   Loading {split_name} from {file_path}")
            
            if not file_path.exists():
                raise FileNotFoundError(f"IOB file not found: {file_path}")
            
            # Load JSON data
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to HuggingFace dataset format with expected field names
            dataset_dict = {
                "words": data["input"],    # Rename 'input' to 'words'
                "ner": data["label"]       # Rename 'label' to 'ner'
            }
            
            # Create Dataset object
            datasets[split_name] = Dataset.from_dict(dataset_dict)
            
            self.logger.info(f"   ✅ {split_name}: {len(datasets[split_name]):,} examples")
        
        return DatasetDict(datasets)
    
    def _validate_dataset(self):
        """Validate dataset structure and consistency."""
        self.logger.info("🔍 Validating dataset structure")
        
        validation_results = validate_dataset_structure(
            self.dataset, 
            label_list=None,  # Will be determined from data
            verbose=True
        )
        
        # Log warnings if any
        if validation_results['warnings']:
            for warning in validation_results['warnings']:
                self.logger.warning(f"⚠️  {warning}")
        
        self.logger.info("✅ Dataset validation completed")
    
    def _initialize_tokenizer(self):
        """Initialize the tokenizer."""
        self.logger.info(f"🔤 Initializing tokenizer: {self.config.model.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.model_name,
                add_prefix_space=True,
                use_fast=self.config.data.use_fast_tokenizer,
                cache_dir=self.config.data.cache_dir
            )
            
            self.logger.info("✅ Tokenizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize tokenizer: {e}")
            raise
    
    def _process_labels(self):
        """Process labels and create mappings."""
        self.logger.info("🏷️  Processing labels and creating mappings")
        
        # Extract all unique labels from the dataset
        all_ner_tags_set = set()
        for example in self.dataset['train']:
            all_ner_tags_set.update(example['ner'])
        
        # Organize labels: O first, then sorted B-/I- tags
        if 'O' in all_ner_tags_set:
            self.label_list = ['O'] + sorted(list(all_ner_tags_set - {'O'}))
        else:
            self.label_list = sorted(list(all_ner_tags_set))
        
        # Create mappings
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        
        # Update config with actual number of labels
        self.config.model.num_labels = len(self.label_list)
        
        self.logger.info(f"✅ Label processing completed")
        self.logger.info(f"   Total labels: {len(self.label_list)}")
        self.logger.info(f"   Label list: {self.label_list[:10]}{'...' if len(self.label_list) > 10 else ''}")
    
    def _tokenize_data(self):
        """Tokenize the dataset with sliding window."""
        self.logger.info("🔤 Tokenizing dataset with sliding window approach")
        
        def tokenize_function(examples):
            return tokenize_and_align_labels(
                examples,
                self.tokenizer,
                max_length=self.config.data.max_length,
                overlap=self.config.data.stride,
                label2id=self.label2id  # Pass the label2id mapping
            )
        
        try:
            self.tokenized_dataset = self.dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=self.dataset["train"].column_names,
                desc="Tokenizing dataset"
            )
            
            # Clean up extra columns that might cause issues
            # Keep only the essential columns for training
            essential_columns = {'input_ids', 'attention_mask', 'labels'}
            for split in self.tokenized_dataset:
                current_columns = set(self.tokenized_dataset[split].column_names)
                columns_to_remove = current_columns - essential_columns
                if columns_to_remove:
                    self.logger.info(f"   Removing extra columns from {split}: {columns_to_remove}")
                    self.tokenized_dataset[split] = self.tokenized_dataset[split].remove_columns(list(columns_to_remove))
            
            # Calculate expansion ratio
            original_size = len(self.dataset['train'])
            new_size = len(self.tokenized_dataset['train'])
            expansion_ratio = new_size / original_size
            
            self.logger.info("✅ Tokenization completed")
            self.logger.info(f"   Original examples: {original_size:,}")
            self.logger.info(f"   Tokenized examples: {new_size:,}")
            self.logger.info(f"   Expansion ratio: {expansion_ratio:.2f}x")
            self.logger.info(f"   Final columns: {self.tokenized_dataset['train'].column_names}")
            
        except Exception as e:
            self.logger.error(f"❌ Tokenization failed: {e}")
            raise
    
    def _apply_augmentation(self):
        """Apply data augmentation for rare classes."""
        self.logger.info("🔄 Applying data augmentation for rare classes")
        
        try:
            self.tokenized_dataset = apply_rare_class_augmentation(
                self.tokenized_dataset,
                enable_augmentation=True,
                multiplier=self.config.data.augmentation_multiplier,
                target_classes=self.config.data.target_rare_classes,
                verbose=True
            )
            
            self.logger.info("✅ Data augmentation completed")
            
        except Exception as e:
            self.logger.error(f"❌ Data augmentation failed: {e}")
            raise
    
    def _compute_class_weights(self):
        """Compute class weights for imbalanced data."""
        self.logger.info("⚖️  Computing class weights for imbalanced data")
        
        try:
            self.class_weights = compute_class_weights(
                self.tokenized_dataset,
                self.label_list,
                method=self.config.data.class_weight_method,
                original_dataset=self.dataset,
                verbose=True
            )
            
            self.logger.info("✅ Class weights computation completed")
            
        except Exception as e:
            self.logger.error(f"❌ Class weights computation failed: {e}")
            raise
    
    def _print_final_statistics(self):
        """Print comprehensive dataset statistics."""
        self.logger.info("📊 Final dataset statistics:")
        
        print_dataset_summary(
            self.tokenized_dataset,
            self.dataset,
            self.label_list
        )


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

class GreekLegalNERModelBuilder:
    """
    Model builder for Greek Legal NER.
    
    Handles model initialization, loss function setup, and trainer creation
    with proper configuration and error handling.
    """
    
    def __init__(self, config: ExperimentConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def build_model_and_trainer(
        self,
        tokenized_dataset,
        tokenizer,
        label_mappings: Dict[str, Any],
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Build model and trainer with complete configuration.
        
        Args:
            tokenized_dataset: Processed dataset
            tokenizer: Initialized tokenizer
            label_mappings: Label mappings (id2label, label2id, label_list)
            class_weights: Class weights for loss function
        
        Returns:
            Tuple of (model, trainer, data_collator)
        """
        
        self.logger.info("🤖 Building model and trainer")
        
        # 1. Initialize model
        model = self._initialize_model(label_mappings)
        
        # Move model to device (CUDA if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        self.logger.info(f"🚀 Model moved to device: {device}")
        
        # 2. Setup loss function
        loss_function = self._setup_loss_function(class_weights, label_mappings['label_list'])
        
        # Move loss function to device if it exists
        if loss_function is not None and hasattr(loss_function, 'to'):
            loss_function = loss_function.to(device)
            self.logger.info(f"🔥 Loss function moved to device: {device}")
        
        # 3. Create data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True
        )
        
        # 4. Setup training arguments
        training_args = self._create_training_arguments()
        
        # 5. Create compute metrics function
        compute_metrics_fn = self._create_compute_metrics_function(
            label_mappings['label_list'], model
        )
        
        # 6. Initialize trainer
        trainer = self._initialize_trainer(
            model, training_args, tokenized_dataset, 
            data_collator, compute_metrics_fn, loss_function,
            label_mappings['label_list']
        )
        
        self.logger.info("✅ Model and trainer initialization completed")
        
        return model, trainer, data_collator
    
    def _initialize_model(self, label_mappings: Dict[str, Any]):
        """Initialize the model (ROBERTa or ROBERTa+CRF)."""
        self.logger.info(f"🏗️  Initializing model: {self.config.model.model_name}")
        
        try:
            # Load configuration
            model_config = AutoConfig.from_pretrained(
                self.config.model.model_name,
                num_labels=self.config.model.num_labels,
                id2label=label_mappings['id2label'],
                label2id=label_mappings['label2id'],
                hidden_dropout_prob=self.config.model.hidden_dropout_prob,
                attention_probs_dropout_prob=self.config.model.attention_dropout_prob
            )
            
            if self.config.model.use_crf:
                # ROBERTa + CRF model
                self.logger.info("   Using ROBERTa + CRF architecture")
                model = BertCRFForTokenClassification(
                    config=model_config,
                    num_labels=self.config.model.num_labels,
                    bert_model_name=self.config.model.model_name
                )
            else:
                # Standard ROBERTa model
                self.logger.info("   Using standard ROBERTa architecture")
                from transformers import AutoModelForTokenClassification
                model = AutoModelForTokenClassification.from_pretrained(
                    self.config.model.model_name,
                    config=model_config
                )
            
            # Apply layer freezing if specified
            if self.config.model.freeze_bert_layers is not None:
                self.logger.info(f"   Freezing first {self.config.model.freeze_bert_layers} layers")
                if hasattr(model, 'freeze_bert_layers'):
                    model.freeze_bert_layers(self.config.model.freeze_bert_layers)
            
            self.logger.info("✅ Model initialized successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {e}")
            raise
    
    def _setup_loss_function(self, class_weights: Optional[torch.Tensor], label_list: List[str]):
        """Setup loss function based on configuration."""
        if self.config.loss.loss_type == "cross_entropy" and not self.config.loss.use_class_weights:
            # Standard cross entropy - will use model's default loss
            return None
        
        self.logger.info(f"🔥 Setting up loss function: {self.config.loss.get_loss_description()}")
        
        try:
            loss_function = create_focal_loss(
                loss_type=self.config.loss.loss_type,
                num_classes=len(label_list),
                class_weights=class_weights if self.config.loss.use_class_weights else None,
                alpha=self.config.loss.focal_alpha,
                gamma=self.config.loss.focal_gamma,
                reduction=self.config.loss.focal_reduction
            )
            
            self.logger.info("✅ Loss function setup completed")
            return loss_function
            
        except Exception as e:
            self.logger.error(f"❌ Loss function setup failed: {e}")
            raise
    
    def _create_training_arguments(self) -> TrainingArguments:
        """Create training arguments from configuration."""
        self.logger.info("⚙️  Creating training arguments")
        
        # Ensure output directory exists
        self.config.output.ensure_directories()
        
        return TrainingArguments(
            output_dir=str(self.config.output.get_experiment_dir()),
            
            # Training schedule
            num_train_epochs=self.config.training.num_train_epochs,
            learning_rate=self.config.training.learning_rate,
            warmup_steps=self.config.training.warmup_steps,
            weight_decay=self.config.training.weight_decay,
            
            # Batch sizes
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            
            # Evaluation and saving
            eval_strategy=self.config.training.evaluation_strategy,  # Changed from evaluation_strategy
            eval_steps=self.config.training.eval_steps,
            save_strategy=self.config.training.save_strategy,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            
            # Logging
            logging_strategy=self.config.training.logging_strategy,
            logging_steps=self.config.training.logging_steps,
            logging_dir=self.config.output.logging_dir,
            
            # Optimization
            adam_epsilon=self.config.training.adam_epsilon,
            max_grad_norm=self.config.training.max_grad_norm,
            
            # Best model selection
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            
            # Performance optimizations
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            dataloader_pin_memory=self.config.training.dataloader_pin_memory,
            
            # Reproducibility
            seed=self.config.training.seed,
            data_seed=self.config.training.data_seed,
            
            # Reporting
            report_to=self.config.training.report_to,
            run_name=self.config.output.run_name,
            
            # Misc
            remove_unused_columns=False,  # Important for token classification
            push_to_hub=False
        )
    
    def _create_compute_metrics_function(self, label_list: List[str], model):
        """Create compute metrics function for trainer."""
        def compute_metrics(eval_pred):
            return compute_metrics_with_model_info(
                eval_pred,
                label_list=label_list,
                model=model,
                use_crf=self.config.model.use_crf,
                verbose=True
            )
        
        return compute_metrics
    
    def _initialize_trainer(
        self, 
        model, 
        training_args, 
        tokenized_dataset,
        data_collator,
        compute_metrics_fn,
        loss_function,
        label_list: List[str]
    ):
        """Initialize the FlexibleTrainer."""
        self.logger.info("🎯 Initializing FlexibleTrainer")
        
        try:
            trainer = FlexibleTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["validation"],  # Use validation for training monitoring
                data_collator=data_collator,
                compute_metrics=compute_metrics_fn,
                
                # FlexibleTrainer specific arguments
                loss_type=self.config.loss.loss_type,
                focal_gamma=self.config.loss.focal_gamma,
                focal_alpha=self.config.loss.focal_alpha,
                use_crf=self.config.model.use_crf,
                label_list=label_list,  # Added label_list parameter
                debug_mode=True  # Added to match old script behavior
            )
            
            self.logger.info("✅ FlexibleTrainer initialized successfully")
            return trainer
            
        except Exception as e:
            self.logger.error(f"❌ Trainer initialization failed: {e}")
            raise


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class GreekLegalNERExperiment:
    """
    Complete experiment runner for Greek Legal NER.
    
    Orchestrates the entire training and evaluation pipeline with
    comprehensive logging, error handling, and result management.
    """
    
    def __init__(self, config: ExperimentConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.results = {}
    
    def run_experiment(self) -> Dict[str, Any]:
        """
        Run complete NER experiment.
        
        🚀 EXPERIMENT PIPELINE:
        ======================
        1. Setup and validation
        2. Data loading and processing
        3. Model and trainer initialization  
        4. Training with monitoring
        5. Evaluation and analysis
        6. Results saving and cleanup
        
        Returns:
            Dict: Comprehensive experiment results
        """
        
        self.logger.info("="*80)
        self.logger.info(f"🚀 STARTING EXPERIMENT: {self.config.name}")
        self.logger.info("="*80)
        
        start_time = datetime.now()
        
        try:
            # 1. Print experiment configuration
            self._print_experiment_info()
            
            # 2. Setup reproducibility
            self._setup_reproducibility()
            
            # 3. Process data
            tokenized_dataset, tokenizer, label_mappings, class_weights = self._process_data()
            
            # 4. Build model and trainer
            model, trainer, data_collator = self._build_model_and_trainer(
                tokenized_dataset, tokenizer, label_mappings, class_weights
            )
            
            # 5. Train model
            training_results = self._train_model(trainer)
            
            # 6. Evaluate model
            evaluation_results = self._evaluate_model(trainer, label_mappings['label_list'], tokenized_dataset["test"])
            
            # 7. Save results and model
            self._save_results_and_model(
                trainer, training_results, evaluation_results, 
                tokenizer, label_mappings
            )
            
            # 8. Compile final results
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.results = {
                'experiment_config': self.config.to_dict(),
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'metadata': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_seconds': duration.total_seconds(),
                    'success': True
                }
            }
            
            self.logger.info("="*80)
            self.logger.info(f"✅ EXPERIMENT COMPLETED SUCCESSFULLY")
            self.logger.info(f"   Duration: {duration}")
            f1_score = evaluation_results.get('eval_f1', 'N/A')
            f1_display = f"{f1_score:.4f}" if isinstance(f1_score, (int, float)) else str(f1_score)
            self.logger.info(f"   Final F1 Score: {f1_display}")
            self.logger.info("="*80)
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ EXPERIMENT FAILED: {e}")
            self.results = {
                'experiment_config': self.config.to_dict(),
                'error': str(e),
                'metadata': {
                    'start_time': start_time.isoformat(),
                    'success': False
                }
            }
            raise
    
    def _print_experiment_info(self):
        """Print experiment configuration summary."""
        self.logger.info("\n" + get_config_summary(self.config))
    
    def _setup_reproducibility(self):
        """Setup reproducibility with seeds."""
        self.logger.info(f"🔧 Setting up reproducibility (seed: {self.config.training.seed})")
        set_seed(self.config.training.seed)
    
    def _process_data(self):
        """Process data using GreekLegalNERDataProcessor."""
        processor = GreekLegalNERDataProcessor(self.config, self.logger)
        return processor.load_and_process_data()
    
    def _build_model_and_trainer(self, tokenized_dataset, tokenizer, label_mappings, class_weights):
        """Build model and trainer using GreekLegalNERModelBuilder."""
        builder = GreekLegalNERModelBuilder(self.config, self.logger)
        return builder.build_model_and_trainer(
            tokenized_dataset, tokenizer, label_mappings, class_weights
        )
    
    def _train_model(self, trainer):
        """Train the model and return training results."""
        self.logger.info("🎯 Starting model training")
        
        try:
            training_output = trainer.train()
            
            self.logger.info("✅ Training completed successfully")
            self.logger.info(f"   Final training loss: {training_output.training_loss:.4f}")
            
            # Extract best model information
            best_model_info = {}
            if hasattr(trainer, 'state') and trainer.state:
                if hasattr(trainer.state, 'best_model_checkpoint') and trainer.state.best_model_checkpoint:
                    best_model_info['best_model_checkpoint'] = trainer.state.best_model_checkpoint
                    # Extract epoch from checkpoint path (e.g., checkpoint-2220 -> epoch calculation)
                    if 'checkpoint-' in trainer.state.best_model_checkpoint:
                        checkpoint_step = trainer.state.best_model_checkpoint.split('checkpoint-')[-1]
                        try:
                            step_num = int(checkpoint_step)
                            # Calculate epoch from step (approximation)
                            steps_per_epoch = len(trainer.get_train_dataloader())
                            best_epoch = step_num / steps_per_epoch
                            best_model_info['best_epoch'] = round(best_epoch, 2)
                            best_model_info['best_step'] = step_num
                            self.logger.info(f"   Best model at epoch: {best_epoch:.2f} (step {step_num})")
                        except (ValueError, ZeroDivisionError):
                            pass
                
                if hasattr(trainer.state, 'best_metric') and trainer.state.best_metric is not None:
                    best_model_info['best_metric'] = trainer.state.best_metric
                    self.logger.info(f"   Best metric value: {trainer.state.best_metric:.4f}")
                
                # Extract training history (epoch-by-epoch metrics)
                if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
                    # Filter out non-epoch entries and extract epoch metrics
                    epoch_history = []
                    for entry in trainer.state.log_history:
                        if 'epoch' in entry and 'eval_loss' in entry:
                            epoch_entry = {
                                'epoch': entry['epoch'],
                                'eval_loss': entry.get('eval_loss'),
                                'eval_overall_f1': entry.get('eval_overall_f1'),
                                'eval_macro_f1': entry.get('eval_macro_f1'),
                                'eval_micro_f1': entry.get('eval_micro_f1'),
                                'train_loss': entry.get('train_loss')
                            }
                            # Remove None values
                            epoch_entry = {k: v for k, v in epoch_entry.items() if v is not None}
                            if len(epoch_entry) > 1:  # More than just epoch
                                epoch_history.append(epoch_entry)
                    
                    if epoch_history:
                        best_model_info['training_history'] = epoch_history
                        # Find best epoch from history
                        best_f1_entry = max(epoch_history, key=lambda x: x.get('eval_overall_f1', 0))
                        if 'eval_overall_f1' in best_f1_entry:
                            best_model_info['best_epoch_from_history'] = best_f1_entry['epoch']
                            best_model_info['best_f1_from_history'] = best_f1_entry['eval_overall_f1']
                            self.logger.info(f"   Best F1 epoch: {best_f1_entry['epoch']} (F1: {best_f1_entry['eval_overall_f1']:.4f})")
            
            return {
                'training_loss': training_output.training_loss,
                'global_step': training_output.global_step,
                'training_time': training_output.metrics.get('train_runtime', 0),
                'training_samples_per_second': training_output.metrics.get('train_samples_per_second', 0),
                'best_model_info': best_model_info
            }
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            raise
    
    def _evaluate_model(self, trainer, label_list, test_dataset):
        """Evaluate the model with comprehensive metrics on test set."""
        self.logger.info("📊 Starting comprehensive model evaluation on test set")
        
        try:
            # Standard evaluation on test set
            eval_results = trainer.evaluate(eval_dataset=test_dataset)
            
            # Additional detailed analysis on test set
            test_predictions = trainer.predict(test_dataset)
            
            # Create detailed evaluation
            calculator = NERMetricsCalculator(label_list, self.config.model.use_crf)
            detailed_results = calculator.compute_comprehensive_metrics(
                test_predictions.predictions,
                test_predictions.label_ids,
                trainer.model,
                verbose=True
            )
            
            self.logger.info("✅ Evaluation completed successfully")
            f1_score = eval_results.get('eval_f1', 'N/A')
            f1_display = f"{f1_score:.4f}" if isinstance(f1_score, (int, float)) else str(f1_score)
            self.logger.info(f"   Test F1 Score: {f1_display}")
            
            # Save only prediction statistics, not the full prediction arrays
            prediction_stats = {
                'num_examples': len(test_predictions.predictions) if hasattr(test_predictions.predictions, '__len__') else None,
                'prediction_shape': test_predictions.predictions.shape if hasattr(test_predictions.predictions, 'shape') else None,
                'metrics_computed': True
            } if test_predictions else {'metrics_computed': False}
            
            return {
                'standard_metrics': eval_results,
                'detailed_metrics': detailed_results.get_summary_dict(),
                'prediction_stats': prediction_stats
            }
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {e}")
            raise
    
    def _save_results_and_model(
        self, 
        trainer, 
        training_results,
        evaluation_results, 
        tokenizer,
        label_mappings
    ):
        """Save model, results, and artifacts."""
        self.logger.info("Saving model and results")
        
        try:
            # Save model and tokenizer
            if self.config.output.save_model:
                model_dir = self.config.output.get_experiment_dir() / "model"
                trainer.save_model(str(model_dir))
                
                if self.config.output.save_tokenizer:
                    tokenizer.save_pretrained(str(model_dir))
                
                # Save label mappings
                with open(model_dir / "label_mappings.json", 'w', encoding='utf-8') as f:
                    clean_label_mappings = clean_for_json_serialization(label_mappings)
                    json.dump(clean_label_mappings, f, indent=2, ensure_ascii=False)
            
            # Save detailed results
            if self.config.output.save_results:
                results_dir = self.config.output.get_experiment_dir() / "results"
                results_dir.mkdir(exist_ok=True)
                
                # Save as JSON
                if "json" in self.config.output.result_formats:
                    results_file = results_dir / f"{self.config.name}_results.json"
                    combined_results = {
                        'config': self.config.to_dict(),
                        'training': training_results,
                        'evaluation': evaluation_results['standard_metrics'],
                        'detailed_metrics': evaluation_results['detailed_metrics'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    with open(results_file, 'w', encoding='utf-8') as f:
                        clean_combined_results = clean_for_json_serialization(combined_results)
                        json.dump(clean_combined_results, f, indent=2, ensure_ascii=False)
                
                # Export detailed evaluation results
                if evaluation_results.get('detailed_metrics'):
                    from evaluation_metrics import EvaluationResults, export_detailed_results
                    
                    # Create EvaluationResults object from summary dict
                    # This is a simplified version - in practice you'd want full object reconstruction
                    for format_type in self.config.output.result_formats:
                        if format_type != "json":
                            output_file = results_dir / f"{self.config.name}_detailed.{format_type}"
                            # Note: This would need full EvaluationResults object reconstruction
                            # For now, just save the summary
            
            self.logger.info("✅ Model and results saved successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Saving failed: {e}")
            raise


# =============================================================================
# MULTI-RUN EXPERIMENT RUNNER (NEW)
# =============================================================================

class MultiRunExperiment:
    """
    Multi-run experiment runner for reliable NER evaluation.
    
    Runs the same experiment multiple times with different seeds and
    aggregates results for statistical significance.
    """
    
    def __init__(self, config: ExperimentConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.individual_results = []
        self.aggregated_results = {}
    
    def run_multi_experiment(self) -> Dict[str, Any]:
        """
        Run multiple experiments with different seeds.
        
        Returns:
            Dict: Aggregated results with statistics
        """
        
        if self.config.num_runs == 1:
            # Single run - use existing single experiment
            self.logger.info("🔧 Single run mode - using standard experiment")
            experiment = GreekLegalNERExperiment(self.config, self.logger)
            return experiment.run_experiment()
        
        # Multi-run mode
        self.logger.info("="*100)
        self.logger.info(f"🚀 STARTING MULTI-RUN EXPERIMENT: {self.config.name}")
        self.logger.info(f"   Number of runs: {self.config.num_runs}")
        self.logger.info(f"   Base seed: {self.config.base_seed or self.config.training.seed}")
        self.logger.info(f"   Seed increment: {self.config.seed_increment}")
        self.logger.info("="*100)
        
        # Setup base seed
        base_seed = self.config.base_seed or self.config.training.seed
        
        # Setup multi-run directory structure
        multirun_dir = self._setup_multirun_directory()
        
        start_time = datetime.now()
        
        # Run individual experiments
        for run_idx in range(self.config.num_runs):
            current_seed = base_seed + (run_idx * self.config.seed_increment)
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"🎯 STARTING RUN {run_idx + 1}/{self.config.num_runs} (seed: {current_seed})")
            self.logger.info(f"{'='*80}")
            
            try:
                run_result = self._run_single_experiment(run_idx + 1, current_seed, multirun_dir)
                self.individual_results.append(run_result)
                
                self.logger.info(f"✅ Run {run_idx + 1} completed successfully")
                
                # Extract F1 score for logging
                f1_score = run_result.get('evaluation_results', {}).get('standard_metrics', {}).get('eval_overall_f1', 'N/A')
                if isinstance(f1_score, (int, float)):
                    self.logger.info(f"   F1 Score: {f1_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"❌ Run {run_idx + 1} failed: {e}")
                
                # Save partial results even on failure
                failed_result = {
                    'run_number': run_idx + 1,
                    'seed': current_seed,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                self.individual_results.append(failed_result)
                
                # Save partial results immediately
                self._save_partial_results(multirun_dir, failed_result)
                
                # Continue with next run if continue_on_failure is True
                if not self.config.continue_on_failure:
                    self.logger.error(f"💥 Stopping multi-run experiment due to failure in run {run_idx + 1}")
                    break
                else:
                    self.logger.warning(f"⚠️ Continuing with next run despite failure in run {run_idx + 1}")
        
        # Aggregate results if we have successful runs
        end_time = datetime.now()
        duration = end_time - start_time
        
        if any(result.get('success', False) for result in self.individual_results):
            self.aggregated_results = self._aggregate_results()
        
        # Save final aggregated results
        final_results = {
            'experiment_config': self.config.to_dict(),
            'num_runs_attempted': self.config.num_runs,
            'num_runs_completed': len([r for r in self.individual_results if r.get('success', False)]),
            'base_seed': base_seed,
            'seed_increment': self.config.seed_increment,
            'individual_results': self.individual_results,
            'aggregated_results': self.aggregated_results,
            'metadata': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'success': len(self.aggregated_results) > 0
            }
        }
        
        self._save_final_results(multirun_dir, final_results)
        
        self.logger.info("="*100)
        self.logger.info(f"✅ MULTI-RUN EXPERIMENT COMPLETED")
        self.logger.info(f"   Duration: {duration}")
        self.logger.info(f"   Successful runs: {final_results['num_runs_completed']}/{final_results['num_runs_attempted']}")
        
        if self.aggregated_results:
            overall_f1_mean = self.aggregated_results.get('overall_f1', {}).get('mean', 'N/A')
            overall_f1_std = self.aggregated_results.get('overall_f1', {}).get('std', 'N/A')
            if isinstance(overall_f1_mean, (int, float)) and isinstance(overall_f1_std, (int, float)):
                self.logger.info(f"   Overall F1: {overall_f1_mean:.4f} ± {overall_f1_std:.4f}")
        
        self.logger.info("="*100)
        
        return final_results
    
    def _setup_multirun_directory(self) -> Path:
        """Setup directory structure for multi-run experiment."""
        multirun_dir = Path(self.config.output.output_dir) / f"{self.config.name}_multirun"
        multirun_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📁 Multi-run directory: {multirun_dir}")
        return multirun_dir
    
    def _run_single_experiment(self, run_number: int, seed: int, multirun_dir: Path) -> Dict[str, Any]:
        """Run a single experiment with specified seed."""
        
        # Create config copy for this run
        run_config = ExperimentConfig.from_dict(self.config.to_dict())
        run_config.training.seed = seed
        run_config.name = f"{self.config.name}_run{run_number}_seed{seed}"
        
        # Setup individual run directory
        run_dir = multirun_dir / f"run_{run_number}_seed{seed}"
        run_config.output.output_dir = str(run_dir)
        run_config.output.experiment_name = run_config.name
        
        # Run experiment
        experiment = GreekLegalNERExperiment(run_config, self.logger)
        result = experiment.run_experiment()
        
        # Add run metadata
        result['run_metadata'] = {
            'run_number': run_number,
            'seed': seed,
            'success': True
        }
        
        return result
    
    def _save_partial_results(self, multirun_dir: Path, failed_result: Dict[str, Any]):
        """Save partial results when a run fails."""
        partial_file = multirun_dir / f"partial_results_run{failed_result['run_number']}.json"
        
        with open(partial_file, 'w', encoding='utf-8') as f:
            clean_result = clean_for_json_serialization(failed_result)
            json.dump(clean_result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Partial results saved: {partial_file}")
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results from successful runs."""
        successful_results = [
            r for r in self.individual_results 
            if r.get('run_metadata', {}).get('success', False) or r.get('success', False)
        ]
        
        if not successful_results:
            return {}
        
        aggregated = {}
        
        # Extract F1 scores and other metrics
        metrics_to_aggregate = [
            'eval_overall_f1', 'eval_overall_precision', 'eval_overall_recall',
            'eval_macro_f1', 'eval_micro_f1', 'eval_tag_accuracy'
        ]
        
        # Overall metrics
        for metric in metrics_to_aggregate:
            values = []
            for result in successful_results:
                eval_metrics = result.get('evaluation_results', {}).get('standard_metrics', {})
                if metric in eval_metrics:
                    values.append(eval_metrics[metric])
            
            if values:
                aggregated[metric.replace('eval_', '')] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': values
                }
        
        # Per-entity F1 scores
        entity_metrics = {}
        
        # Find all entity types from first successful result
        first_result = successful_results[0]
        eval_metrics = first_result.get('evaluation_results', {}).get('standard_metrics', {})
        
        entity_f1_keys = [k for k in eval_metrics.keys() if k.endswith('_f1') and not k.startswith('eval_overall') and not k.startswith('eval_macro') and not k.startswith('eval_micro')]
        
        for entity_key in entity_f1_keys:
            values = []
            for result in successful_results:
                eval_metrics = result.get('evaluation_results', {}).get('standard_metrics', {})
                if entity_key in eval_metrics:
                    values.append(eval_metrics[entity_key])
            
            if values:
                entity_name = entity_key.replace('eval_', '').replace('_f1', '')
                entity_metrics[f"{entity_name}_f1"] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': values
                }
        
        aggregated['entity_metrics'] = entity_metrics
        
        return aggregated
    
    def _save_final_results(self, multirun_dir: Path, final_results: Dict[str, Any]):
        """Save final aggregated results."""
        results_file = multirun_dir / "aggregated_results.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            clean_results = clean_for_json_serialization(final_results)
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Final aggregated results saved: {results_file}")


# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================

def run_single_experiment(config: ExperimentConfig, logger: logging.Logger) -> Dict[str, Any]:
    """
    Run a single or multi-run experiment with the given configuration.
    
    Args:
        config: Experiment configuration
        logger: Logger instance
    
    Returns:
        Dict: Experiment results (single or aggregated)
    """
    
    # Use the new Multi-Run experiment runner which handles both single and multi runs
    multi_runner = MultiRunExperiment(config, logger)
    return multi_runner.run_multi_experiment()


def run_multiple_experiments(
    preset_names: List[str], 
    logger: logging.Logger,
    base_output_dir: str = "./experiments"
) -> Dict[str, Any]:
    """
    Run multiple experiments with different presets.
    
    Args:
        preset_names: List of preset configuration names
        logger: Logger instance
        base_output_dir: Base directory for experiment outputs
    
    Returns:
        Dict: Results from all experiments
    """
    
    all_results = {}
    
    for preset_name in preset_names:
        logger.info(f"\n{'='*100}")
        logger.info(f"🚀 STARTING EXPERIMENT: {preset_name}")
        logger.info(f"{'='*100}")
        
        try:
            # Load preset configuration
            config = ExperimentConfig.get_preset(preset_name)
            
            # Customize output directory for this experiment
            config.output.experiment_name = preset_name.lower()
            config.output.output_dir = base_output_dir
            
            # Run experiment
            results = run_single_experiment(config, logger)
            all_results[preset_name] = results
            
            logger.info(f"✅ {preset_name} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ {preset_name} failed: {e}")
            all_results[preset_name] = {
                'error': str(e),
                'success': False
            }
    
    # Save comparison results
    comparison_file = Path(base_output_dir) / "experiment_comparison.json"
    comparison_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        clean_all_results = clean_for_json_serialization(all_results)
        json.dump(clean_all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Comparison results saved to: {comparison_file}")
    
    return all_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Greek Legal NER Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_script.py --preset BASELINE
  python main_script.py --preset CRF_ENHANCED --output-dir ./my_experiments
  python main_script.py --preset BASELINE CRF_ENHANCED WEIGHTED
  python main_script.py --config custom_config.json
  python main_script.py --list-presets
        """
    )
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--preset', 
        nargs='+',
        choices=['BASELINE', 'CRF_ENHANCED', 'FOCAL_LOSS', 'WEIGHTED', 'AUGMENTED', 'FULL_ENHANCED'],
        help='Preset configuration(s) to use'
    )
    config_group.add_argument(
        '--config',
        type=str,
        help='Custom configuration file (JSON)'
    )
    config_group.add_argument(
        '--list-presets',
        action='store_true',
        help='List available preset configurations'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./experiments',
        help='Base output directory for experiments'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (optional)'
    )
    
    # Device options
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to use for training'
    )
    
    # Multi-run options (NEW)
    parser.add_argument(
        '--num-runs',
        type=int,
        default=1,
        help='Number of runs with different seeds (default: 1)'
    )
    parser.add_argument(
        '--base-seed',
        type=int,
        help='Starting seed for multi-run (default: use config seed)'
    )
    parser.add_argument(
        '--seed-increment',
        type=int,
        default=1,
        help='Increment per run (default: 1)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Handle list presets
    if args.list_presets:
        list_available_presets()
        return
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_file
    if not log_file and args.output_dir:
        log_dir = Path(args.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / f"training_{timestamp}.log")
    
    logger = setup_logging(args.log_level, log_file)
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"🖥️  Using device: {device}")
    
    try:
        if args.preset:
            # Run preset experiments
            if len(args.preset) == 1:
                # Single experiment
                config = ExperimentConfig.get_preset(args.preset[0])
                config.output.output_dir = args.output_dir
                config.output.experiment_name = args.preset[0].lower()
                
                # Apply multi-run parameters from command line
                if hasattr(args, 'num_runs') and args.num_runs:
                    config.num_runs = args.num_runs
                if hasattr(args, 'base_seed') and args.base_seed is not None:
                    config.base_seed = args.base_seed
                if hasattr(args, 'seed_increment') and args.seed_increment:
                    config.seed_increment = args.seed_increment
                
                results = run_single_experiment(config, logger)
                
            else:
                # Multiple experiments
                results = run_multiple_experiments(
                    args.preset, 
                    logger, 
                    args.output_dir
                )
        
        elif args.config:
            # Load custom configuration
            with open(args.config, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            config = ExperimentConfig.from_dict(config_dict)
            
            # Override output directory if specified
            if args.output_dir != './experiments':
                config.output.output_dir = args.output_dir
            
            # Apply multi-run parameters from command line
            if hasattr(args, 'num_runs') and args.num_runs:
                config.num_runs = args.num_runs
            if hasattr(args, 'base_seed') and args.base_seed is not None:
                config.base_seed = args.base_seed
            if hasattr(args, 'seed_increment') and args.seed_increment:
                config.seed_increment = args.seed_increment
            
            results = run_single_experiment(config, logger)
        
        logger.info("🎉 All experiments completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
