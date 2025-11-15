from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data" / "chestxray14"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)

# =============================================================================
# LABELS (14 pathologies in ChestX-ray14)
# =============================================================================
LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]
NUM_CLASSES = len(LABELS)

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
CONFIG = {
    # -------------------------------------------------------------------------
    # Data Settings
    # -------------------------------------------------------------------------
    "subset_size": 10000,          # Total images to use (None = full dataset)
    "image_size": 224,            # Input image size for model
    "batch_size": 32,             # Batch size for training
    "num_workers": 4,             # DataLoader workers (reduce if memory issues)

    # Split ratios (must sum to 1.0)
    "train_ratio": 0.70,
    "val_ratio": 0.10,
    "cal_ratio": 0.10,
    "test_ratio": 0.10,

    # -------------------------------------------------------------------------
    # Model Settings
    # -------------------------------------------------------------------------
    "model_name": "resnet50",     # Backbone architecture
    "pretrained": True,           # Use ImageNet pretrained weights
    "dropout": 0.5,               # Dropout rate before final layer

    # -------------------------------------------------------------------------
    # Training Settings (2-phase training)
    # -------------------------------------------------------------------------
    # Phase 1: Train only classification head (backbone frozen)
    "phase1_epochs": 30,
    "phase1_lr": 1e-3,

    # Phase 2: Fine-tune backbone layers 3-4
    "phase2_epochs": 10,
    "phase2_lr": 1e-4,

    # Shared training params
    "weight_decay": 1e-5,
    "early_stopping_patience": 5,
    "gradient_clip_norm": 1.0,

    # -------------------------------------------------------------------------
    # Conformal Prediction Settings
    # -------------------------------------------------------------------------
    "alpha": 0.10,                # Miscoverage rate (1-alpha = 90% coverage)

    # CWCS-specific (your method)
    "cwcs_lambda": 1.0,           # Co-occurrence weighting strength
    "cwcs_min_cooccurrence": 0.1, # Minimum threshold for edge inclusion

    # -------------------------------------------------------------------------
    # Device & Reproducibility
    # -------------------------------------------------------------------------
    "device": "mps",              # "mps" for M3 Pro, "cuda" for NVIDIA, "cpu"
    "seed": 2024,                 # Random seed for reproducibility

    # -------------------------------------------------------------------------
    # Evaluation Thresholds
    # -------------------------------------------------------------------------
    "target_coverage": 0.90,      # Target coverage for conformal methods
    "acceptable_f1": 0.40,        # Minimum acceptable macro-F1 for classifier
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_subset_config(subset_size: int = None):
    """
    Get a copy of CONFIG with modified subset size.
    Useful for quick testing vs full experiments.

    Examples:
        config = get_subset_config(1000)   # Quick test with 1K images
        config = get_subset_config(None)   # Full dataset
    """
    config = CONFIG.copy()
    config["subset_size"] = subset_size
    return config


def print_config():
    """Pretty print current configuration"""
    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)

    sections = {
        "Data": ["subset_size", "image_size", "batch_size", "num_workers"],
        "Training": ["phase1_epochs", "phase2_epochs", "phase1_lr", "phase2_lr"],
        "Conformal": ["alpha", "target_coverage", "cwcs_lambda"],
        "System": ["device", "seed"],
    }

    for section, keys in sections.items():
        print(f"\n{section}:")
        for key in keys:
            if key in CONFIG:
                print(f"  {key}: {CONFIG[key]}")

    print("\n" + "=" * 60)


# Print config when module is imported (optional, comment out if noisy)
if __name__ == "__main__":
    print_config()