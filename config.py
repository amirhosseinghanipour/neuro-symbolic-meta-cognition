import torch
import os
from datetime import datetime, timedelta

class Config:
    """
    Configuration parameters for the Neuro-Symbolic Meta-Cognitive system.

    Stores settings for models, reasoning, meta-cognition, rule generation,
    data paths, and training loop parameters.
    """

    # --- Model Configuration ---
    BASE_BERT_MODEL: str = 'bert-base-uncased'
    """Base BERT model for embeddings."""
    NER_MODEL: str = 'dslim/bert-base-NER'
    """NER model for entity recognition pipeline."""
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Device (CPU or CUDA GPU) for PyTorch models."""
    MAX_LENGTH: int = 128
    """Maximum sequence length for tokenization."""

    # --- Symbolic Reasoner Configuration ---
    MAX_REASONING_DEPTH: int = 5
    """Maximum forward-chaining iterations."""
    INITIAL_RULE_CONFIDENCE: float = 0.9
    """Default confidence for new rules."""
    MIN_RULE_CONFIDENCE_THRESHOLD: float = 0.1
    """Minimum confidence for a rule to be considered active during reasoning."""

    # --- Meta-Cognition Configuration ---
    PLAUSIBILITY_THRESHOLD: float = 0.5
    """Confidence threshold for accepting an answer as plausible."""
    ADAPTATION_LEARNING_RATE_NEURAL: float = 2e-5
    """Learning rate for fine-tuning the neural component."""
    ADAPTATION_LEARNING_RATE_SYMBOLIC: float = 0.1
    """Step size for adjusting symbolic rule confidences."""
    NEURAL_WEIGHT_DECAY: float = 0.01
    """Weight decay for the AdamW optimizer."""

    # --- Rule Generation Configuration ---
    RULE_GEN_MIN_SUPPORT: int = 2
    """Minimum similar failures to trigger batch rule generation."""
    RULE_GEN_CLUSTER_THRESHOLD: float = 0.85
    """Cosine similarity threshold for clustering failed questions."""

    # --- Data Configuration ---
    DATA_DIR: str = "data"
    SAMPLE_QA_FILENAME: str = "sample_qa.json"
    DATA_PATH: str = os.path.join(DATA_DIR, SAMPLE_QA_FILENAME)

    # --- Training Loop Configuration ---
    NUM_EPOCHS: int = 3
    """Number of epochs to run the evaluation/adaptation loop."""
    NEURAL_TUNING_BATCH_SIZE: int = 8
    """Batch size for the neural fine-tuning gradient updates."""
    RUN_FINE_TUNING_AFTER_EPOCH: bool = True
    """Whether to run a neural fine-tuning phase after each evaluation epoch."""

config = Config()
