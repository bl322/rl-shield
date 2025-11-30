import torch

class Config:
    # Random Seed
    SEED = 42
    
    # Action Space Bounds (Paper Section III)
    TEMP_MIN = 0.1
    TEMP_MAX = 2.0
    TOP_P_MIN = 0.1
    TOP_P_MAX = 1.0
    TOP_K_MAX = 100
    
    # PPO Hyperparameters (Paper Algorithm 1)
    LR = 3e-4
    GAMMA = 0.99        # Discount factor
    GAE_LAMBDA = 0.95   # GAE parameter
    CLIP_EPS = 0.2
    EPOCHS = 10         # Iterations per update
    BATCH_SIZE = 64     # For demo; Paper uses 1024
    
    # Model Configuration
    SBERT_MODEL = "all-MiniLM-L6-v2" # Lightweight Sentence-BERT
    HIDDEN_DIM = 128
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default Initial State
    # Normalized values for T=0.7, p=0.9, k=50
    DEFAULT_H_PREV = [0.7, 0.9, 50/100.0]import torch

class Config:
    # Random Seed
    SEED = 42
    
    # Action Space Bounds (Paper Section III)
    TEMP_MIN = 0.1
    TEMP_MAX = 2.0
    TOP_P_MIN = 0.1
    TOP_P_MAX = 1.0
    TOP_K_MAX = 100
    
    # PPO Hyperparameters (Paper Algorithm 1)
    LR = 3e-4
    GAMMA = 0.99        # Discount factor
    GAE_LAMBDA = 0.95   # GAE parameter
    CLIP_EPS = 0.2
    EPOCHS = 10         # Iterations per update
    BATCH_SIZE = 64     # For demo; Paper uses 1024
    
    # Model Configuration
    SBERT_MODEL = "all-MiniLM-L6-v2" # Lightweight Sentence-BERT
    HIDDEN_DIM = 128
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default Initial State
    # Normalized values for T=0.7, p=0.9, k=50
    DEFAULT_H_PREV = [0.7, 0.9, 50/100.0]
