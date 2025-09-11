"""
KolamAI Configuration Management
"""
from dataclasses import dataclass
from typing import Tuple
import os

@dataclass
class DataConfig:
    """Data-related configuration"""
    real_images_dir: str = "data/kolam_images"
    synthetic_count: int = 1000
    image_size: int = 256
    batch_size: int = 16
    
@dataclass
class ModelConfig:
    """Model architecture configuration"""
    latent_dim: int = 100
    generator_features: int = 64
    discriminator_features: int = 64
    num_classes: int = 13
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 200
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    save_interval: int = 25
    
@dataclass
class UIConfig:
    """UI configuration"""
    page_title: str = "KolamAI - Advanced Pattern Analysis & Generation"
    page_icon: str = "🕸️"
    layout: str = "wide"
    
class Config:
    """Main configuration class"""
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.ui = UIConfig()
        self.device = "cuda" if os.system("nvidia-smi") == 0 else "cpu"

# Global config instance
config = Config()

