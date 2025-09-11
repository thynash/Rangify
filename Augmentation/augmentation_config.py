
"""
KolamAI/data/augmentation_config.py
Configuration for Advanced Kolam Augmentation
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import multiprocessing as mp

@dataclass
class AugmentationConfig:
    """Configuration for Kolam augmentation parameters"""

    # Directory settings
    input_dir: str = "data"
    output_dir: str = "augment"

    # Augmentation targets
    target_total_images: int = 107000
    original_image_count: int = 107

    # Processing settings
    num_workers: int = min(mp.cpu_count(), 8)
    batch_size: int = 100

    # Image quality settings
    output_format: str = "jpg"
    jpeg_quality: int = 95
    output_size: Tuple[int, int] = (256, 256)

    # Augmentation probabilities
    geometric_prob: float = 0.7
    photometric_prob: float = 0.6
    noise_prob: float = 0.3
    artistic_prob: float = 0.15

    # Cultural preservation settings
    preserve_symmetry: bool = True
    preserve_cultural_rules: bool = True
    symmetry_enforcement_prob: float = 0.3
    structure_preservation_threshold: float = 0.3

    # Specific augmentation parameters
    rotation_limits: Tuple[int, int] = (-10, 10)
    brightness_limits: Tuple[float, float] = (-0.15, 0.15)
    contrast_limits: Tuple[float, float] = (-0.15, 0.15)
    scale_limits: Tuple[float, float] = (0.9, 1.1)

    # Validation settings
    min_success_rate: float = 0.95
    max_failed_per_image: int = 50

    # Output naming
    output_prefix: str = "kolam"
    output_suffix: str = ""
    filename_padding: int = 6

    def get_images_per_original(self) -> int:
        """Calculate how many augmented images per original"""
        return self.target_total_images // self.original_image_count

    def get_output_filename(self, index: int) -> str:
        """Generate standardized output filename"""
        base_name = f"{self.output_prefix}_{index:0{self.filename_padding}d}"
        if self.output_suffix:
            base_name += f"_{self.output_suffix}"
        return f"{base_name}.{self.output_format}"

# Global config instance
aug_config = AugmentationConfig()

