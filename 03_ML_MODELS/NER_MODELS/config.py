"""
Configuration Management for Greek Legal NER

This module contains all configuration settings for the Greek Legal NER project.
It provides centralized configuration management with different presets for
experiments, model settings, training parameters, and data processing options.

🔧 CONFIGURATION ARCHITECTURE:
==============================

1. **BaseConfig**: Κοινές ρυθμίσεις για όλα τα experiments
2. **ModelConfig**: Model-specific settings (ROBERTa, CRF, etc.)
3. **TrainingConfig**: Training hyperparameters και strategies
4. **DataConfig**: Data processing και augmentation settings
5. **ExperimentConfig**: Preset configurations για διαφορετικά experiments

📊 EXPERIMENT PRESETS:
=====================
- BASELINE: Απλό ROBERTa + CrossEntropy
- CRF_ENHANCED: ROBERTa + CRF
- FOCAL_LOSS: ROBERTa + Focal Loss για imbalanced data
- WEIGHTED: ROBERTa + Class Weights
- FULL_ENHANCED: ROBERTa + CRF + Focal Loss + Weights + Augmentation

🎯 USAGE EXAMPLES:
=================
```python
# Load preset configuration
config = ExperimentConfig.get_preset("CRF_ENHANCED")

# Customize specific settings
config.training.learning_rate = 3e-5
config.data.enable_augmentation = True

# Create trainer with configuration
trainer = FlexibleTrainer(config=config)
```
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from pathlib import Path


# =============================================================================
# BASE CONFIGURATION CLASSES
# =============================================================================

@dataclass
class ModelConfig:
    """
    Model architecture and initialization settings.
    
    🔍 MODEL SELECTION STRATEGY:
    ============================
    - **Primary Model**: AI-team-UoA/GreekLegalRoBERTa_v2
      - Εξειδικευμένο για ελληνικά νομικά κείμενα
      - Pre-trained στο Greek Legal corpus
      - Καλύτερη απόδοση από generic Greek models
    
    - **CRF Integration**: Optional CRF layer για sequence constraints
      - Εφαρμόζει BIO tagging rules
      - Βελτιώνει consistency των predictions
      - Μικρό computational overhead
    
    - **Dropout Strategy**: Balanced approach για regularization
      - 0.1 για training stability
      - Μπορεί να αυξηθεί (0.2-0.3) για overfitting prevention
    """
    
    # Base model settings
    model_name: str = "AI-team-UoA/GreekLegalRoBERTa_v2"
    num_labels: int = 17  # Updated for local IOB format (8 entities × 2 + O = 17)
    use_crf: bool = False
    
    # Model architecture
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    classifier_dropout: float = 0.1
    
    # Weight initialization
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    
    # CRF specific settings
    crf_reduction: str = "mean"  # "mean", "sum", "none"
    
    # Fine-tuning strategy
    freeze_bert_layers: Optional[int] = None  # None = don't freeze, int = freeze first N layers
    gradient_checkpointing: bool = False  # Memory optimization for large models
    
    def validate(self):
        """Validate model configuration settings."""
        if self.num_labels < 2:
            raise ValueError("num_labels must be at least 2")
        if not (0.0 <= self.hidden_dropout_prob <= 1.0):
            raise ValueError("dropout probabilities must be between 0 and 1")
        if self.freeze_bert_layers is not None and self.freeze_bert_layers < 0:
            raise ValueError("freeze_bert_layers must be non-negative")


@dataclass
class TrainingConfig:
    """
    Training hyperparameters and optimization settings.
    
    🎯 TRAINING STRATEGY EXPLANATION:
    ================================
    
    **Learning Rate Schedule:**
    - 5e-5: Conservative για fine-tuning
    - Linear decay με warmup για stability
    - AdamW optimizer με weight decay για regularization
    
    **Batch Size Strategy:**
    - Train: 8 (memory constraints με 512 token sequences)
    - Eval: 16 (μπορεί να είναι μεγαλύτερο χωρίς gradients)
    - Gradient accumulation για effective larger batches
    
    **Epochs & Early Stopping:**
    - 6 epochs: Επαρκή για convergence χωρίς overfitting
    - Early stopping patience: 2 για automatic termination
    - Save best model based on validation F1
    """
    
    # Optimization settings
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0
    
    # Training schedule
    num_train_epochs: int = 6
    warmup_steps: int = 500
    warmup_ratio: float = 0.1  # Alternative to warmup_steps
    
    # Batch settings (Matching old script for exact comparison)
    per_device_train_batch_size: int = 8   # Matching old script TRAIN_BATCH_SIZE
    per_device_eval_batch_size: int = 16   # Matching old script EVAL_BATCH_SIZE
    gradient_accumulation_steps: int = 1
    
    # Evaluation and saving
    evaluation_strategy: str = "epoch"  # "steps", "epoch", "no"
    eval_steps: Optional[int] = None  # Required if evaluation_strategy="steps"
    save_strategy: str = "epoch"
    save_steps: Optional[int] = None
    save_total_limit: int = 3
    
    # Early stopping and monitoring
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_overall_f1"  # Changed to match new evaluation system
    greater_is_better: bool = True
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.001
    
    # Logging and monitoring
    logging_strategy: str = "steps"
    logging_steps: int = 50
    report_to: List[str] = field(default_factory=lambda: [])  # ["wandb", "tensorboard"]
    
    # Reproducibility (ALIGNED WITH OLD SCRIPT)
    seed: int = 44  # Changed from 42 to match old script
    data_seed: Optional[int] = None
    
    # Performance optimizations
    fp16: bool = True  # Mixed precision training for RTX 4060 (faster + less memory)
    bf16: bool = False  # Brain float 16 (if supported)
    dataloader_num_workers: int = 2  # Increased for faster data loading (Windows compatible)
    dataloader_pin_memory: bool = True
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size including gradient accumulation."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps
    
    def validate(self):
        """Validate training configuration settings."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.num_train_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("Batch size must be positive")


@dataclass
class DataConfig:
    """
    Data processing and augmentation settings.
    
    🔍 DATA PROCESSING PIPELINE:
    ===========================
    
    **Tokenization Strategy:**
    - Max length: 512 tokens (ROBERTa limit)
    - Sliding window με 50 token overlap
    - Handles long legal documents efficiently
    
    **Label Alignment:**
    - Subword tokenization handling
    - -100 για ignored tokens (padding, subwords)
    - Preserve entity boundaries correctly
    
    **Class Balancing:**
    - Multiple weighting strategies available
    - Rare class augmentation με intelligent selection
    - Conservative multipliers για stability
    """
    
    # Dataset settings
    dataset_name: str = "local_iob"  # Changed to use local IOB files by default
    dataset_config: Optional[str] = None
    
    # Tokenization settings
    max_length: int = 512
    stride: int = 50  # Overlap for sliding window
    padding: str = "max_length"
    truncation: bool = True
    return_overflowing_tokens: bool = True
    
    # Label processing
    label_all_tokens: bool = False  # Whether to label all tokens or just first subword
    ignore_label_id: int = -100
    
    # Class balancing
    compute_class_weights: bool = True
    class_weight_method: str = "capped_sqrt_inv_freq"  # See data_utils.py for options
    
    # Data augmentation
    enable_augmentation: bool = False
    augmentation_multiplier: int = 3
    target_rare_classes: List[str] = field(default_factory=lambda: [
        "NATIONAL_LOCATION", "UNKNOWN_LOCATION", "FACILITY"  # Updated entity names
    ])
    
    # Validation and testing
    validation_split: float = 0.1  # If no validation set exists
    test_split: float = 0.1  # If no test set exists
    shuffle_train: bool = True
    
    # Caching and performance
    cache_dir: Optional[str] = None
    use_fast_tokenizer: bool = True
    
    def validate(self):
        """Validate data configuration settings."""
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.stride < 0:
            raise ValueError("stride cannot be negative")
        if self.stride >= self.max_length:
            raise ValueError("stride should be less than max_length")
        if not (0.0 <= self.validation_split <= 1.0):
            raise ValueError("validation_split must be between 0 and 1")


@dataclass
class LossConfig:
    """
    Loss function configuration for different training strategies.
    
    🔥 LOSS FUNCTION STRATEGIES:
    ===========================
    
    **Standard CrossEntropy:**
    - Απλή και σταθερή
    - Καλή baseline για comparison
    - Δεν χειρίζεται class imbalance
    
    **Focal Loss:**
    - Εστιάζει σε hard examples
    - Γάμμα=1.0: Conservative approach
    - Άλφα=0.25: Slight boost για rare classes
    
    **Adaptive Focal Loss:**
    - Βελτιωμένη έκδοση για NER
    - Καλύτερο error handling
    - Conservative γάμμα για stability
    
    **Class Weights:**
    - Αντιμετωπίζει class imbalance
    - Συνδυάζεται με οποιοδήποτε loss
    - Multiple computation methods
    """
    
    # Loss type selection
    loss_type: str = "standard_ce"  # "standard_ce", "weighted_ce", "adaptive_focal", "balanced_focal"
    
    # Focal loss parameters
    focal_alpha: Optional[float] = 0.25  # Class balancing factor
    focal_gamma: float = 1.0  # Focusing parameter (conservative)
    focal_reduction: str = "mean"  # "mean", "sum", "none"
    
    # Class weighting
    use_class_weights: bool = False
    class_weight_method: str = "capped_sqrt_inv_freq"
    
    # Loss combination (if multiple losses)
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "main": 1.0,
        "auxiliary": 0.1
    })
    
    def get_loss_description(self) -> str:
        """Get human-readable description of loss configuration."""
        desc = f"Loss: {self.loss_type}"
        if self.loss_type in ["focal", "adaptive_focal"]:
            desc += f" (γ={self.focal_gamma}, α={self.focal_alpha})"
        if self.use_class_weights:
            desc += f" + weights ({self.class_weight_method})"
        return desc


@dataclass
class OutputConfig:
    """
    Output directories and logging configuration.
    """
    
    # Base directories
    output_dir: str = "./results"
    logging_dir: str = "./logs"
    cache_dir: str = "./cache"
    
    # Experiment naming
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    
    # Saving options
    save_model: bool = True
    save_tokenizer: bool = True
    save_results: bool = True
    save_predictions: bool = False
    
    # Result formats
    result_formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    
    def get_experiment_dir(self) -> Path:
        """Get the full experiment directory path."""
        base_dir = Path(self.output_dir)
        if self.experiment_name:
            return base_dir / self.experiment_name
        return base_dir
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        dirs_to_create = [
            self.output_dir,
            self.logging_dir,
            self.cache_dir
        ]
        
        if self.experiment_name:
            dirs_to_create.append(str(self.get_experiment_dir()))
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# =============================================================================
# COMPLETE EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    Complete configuration for a Greek Legal NER experiment.
    
    Combines all configuration components and provides preset configurations
    for common experimental setups.
    """
    
    # Configuration components
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Experiment metadata
    name: str = "greek_legal_ner_experiment"
    description: str = "Greek Legal NER training experiment"
    tags: List[str] = field(default_factory=list)
    
    # Multi-run configuration (NEW)
    num_runs: int = 1                    # Number of runs (1 = single run, >1 = multi-run)
    base_seed: Optional[int] = None      # Starting seed (if None, uses training.seed)
    seed_increment: int = 1              # Increment per run (default: +1)
    aggregate_results: bool = True       # Whether to compute aggregated statistics
    save_individual_models: bool = True  # Save all individual models
    continue_on_failure: bool = False    # Continue if one run fails (always False per request)
    
    def validate(self):
        """Validate all configuration components."""
        self.model.validate()
        self.training.validate()
        self.data.validate()
        
        # Cross-component validation
        if self.loss.use_class_weights and not self.data.compute_class_weights:
            raise ValueError("Class weights requested but not computed in data config")
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization."""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        # Handle nested dataclasses properly
        kwargs = {}
        
        for field in cls.__dataclass_fields__:
            if field in config_dict:
                field_type = cls.__dataclass_fields__[field].type
                value = config_dict[field]
                
                # Handle nested dataclasses
                if field == 'model' and isinstance(value, dict):
                    kwargs[field] = ModelConfig(**value)
                elif field == 'training' and isinstance(value, dict):
                    kwargs[field] = TrainingConfig(**value)
                elif field == 'data' and isinstance(value, dict):
                    kwargs[field] = DataConfig(**value)
                elif field == 'loss' and isinstance(value, dict):
                    kwargs[field] = LossConfig(**value)
                elif field == 'output' and isinstance(value, dict):
                    kwargs[field] = OutputConfig(**value)
                else:
                    kwargs[field] = value
        
        return cls(**kwargs)
    
    @classmethod
    def get_preset(cls, preset_name: str) -> 'ExperimentConfig':
        """
        Get predefined configuration presets.
        
        🎯 AVAILABLE PRESETS:
        ====================
        
        **BASELINE**: Simple ROBERTa + CrossEntropy
        - Standard training setup
        - No special loss functions
        - Good starting point
        
        **CRF_ENHANCED**: ROBERTa + CRF
        - Structured prediction
        - BIO constraint enforcement
        - Better sequence coherence
        
        **FOCAL_LOSS**: ROBERTa + Focal Loss
        - Addresses class imbalance
        - Conservative parameters
        - Focuses on hard examples
        
        **WEIGHTED**: ROBERTa + Class Weights
        - Manual class balancing
        - Capped square root inverse frequency
        - Boosts rare classes
        
        **AUGMENTED**: ROBERTa + Data Augmentation
        - Oversampling rare classes
        - Conservative multiplier
        - Improved rare class recall
        
        **FULL_ENHANCED**: All techniques combined
        - CRF + Focal Loss + Weights + Augmentation
        - Maximum performance approach
        - Careful parameter tuning required
        """
        
        if preset_name == "BASELINE":
            return cls._get_baseline_config()
        elif preset_name == "CRF_ENHANCED":
            return cls._get_crf_config()
        elif preset_name == "FOCAL_LOSS":
            return cls._get_focal_config()
        elif preset_name == "WEIGHTED":
            return cls._get_weighted_config()
        elif preset_name == "AUGMENTED":
            return cls._get_augmented_config()
        elif preset_name == "FULL_ENHANCED":
            return cls._get_full_enhanced_config()
        else:
            raise ValueError(f"Unknown preset: {preset_name}")
    
    @classmethod
    def _get_baseline_config(cls) -> 'ExperimentConfig':
        """Baseline configuration: Simple ROBERTa + CrossEntropy."""
        config = cls()
        config.name = "baseline"
        config.description = "Baseline ROBERTa with standard CrossEntropy loss"
        config.tags = ["baseline", "roberta", "cross_entropy"]
        
        # Model: Standard ROBERTa
        config.model.use_crf = False
        
        # Loss: Standard CrossEntropy
        config.loss.loss_type = "standard_ce"
        config.loss.use_class_weights = False
        
        # Data: No augmentation
        config.data.enable_augmentation = False
        config.data.compute_class_weights = False
        
        return config
    
    @classmethod
    def _get_crf_config(cls) -> 'ExperimentConfig':
        """CRF enhanced configuration."""
        config = cls()
        config.name = "crf_enhanced"
        config.description = "ROBERTa with CRF for structured prediction"
        config.tags = ["crf", "roberta", "structured_prediction"]
        
        # Model: ROBERTa + CRF
        config.model.use_crf = True
        
        # Loss: Standard (CRF handles its own loss)
        config.loss.loss_type = "standard_ce"
        config.loss.use_class_weights = False
        
        return config
    
    @classmethod
    def _get_focal_config(cls) -> 'ExperimentConfig':
        """Focal loss configuration for imbalanced data."""
        config = cls()
        config.name = "focal_loss"
        config.description = "ROBERTa with Focal Loss for imbalanced classes"
        config.tags = ["focal_loss", "roberta", "imbalanced"]
        
        # Model: Standard ROBERTa
        config.model.use_crf = False
        
        # Loss: Adaptive Focal Loss
        config.loss.loss_type = "adaptive_focal"
        config.loss.focal_gamma = 1.0  # Conservative
        config.loss.focal_alpha = 0.25
        config.loss.use_class_weights = False
        
        return config
    
    @classmethod
    def _get_weighted_config(cls) -> 'ExperimentConfig':
        """Class weighted configuration."""
        config = cls()
        config.name = "weighted"
        config.description = "ROBERTa with class weights for imbalanced data"
        config.tags = ["weighted", "roberta", "class_weights"]
        
        # Model: Standard ROBERTa
        config.model.use_crf = False
        
        # Loss: CrossEntropy with class weights
        config.loss.loss_type = "weighted_ce"
        config.loss.use_class_weights = True
        config.loss.class_weight_method = "capped_sqrt_inv_freq"
        
        # Data: Compute class weights
        config.data.compute_class_weights = True
        config.data.class_weight_method = "capped_sqrt_inv_freq"
        
        return config
    
    @classmethod
    def _get_augmented_config(cls) -> 'ExperimentConfig':
        """Data augmentation configuration."""
        config = cls()
        config.name = "augmented"
        config.description = "ROBERTa with data augmentation for rare classes"
        config.tags = ["augmented", "roberta", "oversampling"]
        
        # Model: Standard ROBERTa
        config.model.use_crf = False
        
        # Data: Enable augmentation
        config.data.enable_augmentation = True
        config.data.augmentation_multiplier = 3
        
        return config
    
    @classmethod
    def _get_full_enhanced_config(cls) -> 'ExperimentConfig':
        """Full enhanced configuration with all techniques."""
        config = cls()
        config.name = "full_enhanced"
        config.description = "ROBERTa + CRF + Focal Loss + Weights + Augmentation"
        config.tags = ["enhanced", "crf", "focal_loss", "weighted", "augmented"]
        
        # Model: ROBERTa + CRF
        config.model.use_crf = True
        
        # Loss: Adaptive Focal Loss with class weights
        config.loss.loss_type = "adaptive_focal"
        config.loss.focal_gamma = 1.0
        config.loss.focal_alpha = 0.25
        config.loss.use_class_weights = True
        config.loss.class_weight_method = "capped_sqrt_inv_freq"
        
        # Data: Class weights + augmentation
        config.data.compute_class_weights = True
        config.data.class_weight_method = "capped_sqrt_inv_freq"
        config.data.enable_augmentation = True
        config.data.augmentation_multiplier = 3
        
        return config


# =============================================================================
# PRESET CONFIGURATIONS REGISTRY
# =============================================================================

# Available preset configurations
AVAILABLE_PRESETS = {
    "BASELINE": "Simple ROBERTa + CrossEntropy (baseline)",
    "CRF_ENHANCED": "ROBERTa + CRF for structured prediction",
    "FOCAL_LOSS": "ROBERTa + Focal Loss for imbalanced data",
    "WEIGHTED": "ROBERTa + Class Weights for rare classes",
    "AUGMENTED": "ROBERTa + Data Augmentation for rare classes",
    "FULL_ENHANCED": "All techniques: CRF + Focal + Weights + Augmentation"
}


def list_available_presets():
    """Print all available configuration presets."""
    print("🔧 AVAILABLE CONFIGURATION PRESETS:")
    print("="*50)
    for preset_name, description in AVAILABLE_PRESETS.items():
        print(f"  {preset_name:<15}: {description}")
    print("="*50)


def get_config_summary(config: ExperimentConfig) -> str:
    """Get a human-readable summary of the configuration."""
    summary = []
    summary.append(f"📊 EXPERIMENT: {config.name}")
    summary.append(f"📝 Description: {config.description}")
    summary.append(f"🏷️  Tags: {', '.join(config.tags)}")
    summary.append("")
    summary.append(f"🤖 MODEL:")
    summary.append(f"   Base: {config.model.model_name}")
    summary.append(f"   CRF: {'✅' if config.model.use_crf else '❌'}")
    summary.append(f"   Dropout: {config.model.hidden_dropout_prob}")
    summary.append("")
    summary.append(f"🔥 LOSS:")
    summary.append(f"   {config.loss.get_loss_description()}")
    summary.append("")
    summary.append(f"📚 DATA:")
    summary.append(f"   Max Length: {config.data.max_length}")
    summary.append(f"   Augmentation: {'✅' if config.data.enable_augmentation else '❌'}")
    summary.append(f"   Class Weights: {'✅' if config.data.compute_class_weights else '❌'}")
    summary.append("")
    summary.append(f"🎯 TRAINING:")
    summary.append(f"   LR: {config.training.learning_rate}")
    summary.append(f"   Epochs: {config.training.num_train_epochs}")
    summary.append(f"   Batch Size: {config.training.per_device_train_batch_size}")
    
    return "\n".join(summary)


# Export main classes and functions
__all__ = [
    'ModelConfig',
    'TrainingConfig', 
    'DataConfig',
    'LossConfig',
    'OutputConfig',
    'ExperimentConfig',
    'AVAILABLE_PRESETS',
    'list_available_presets',
    'get_config_summary'
]
