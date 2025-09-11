
"""
KolamAI/data/augmentation.py
Advanced Data Augmentation Module for Kolam Images
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import os
import random
from typing import List, Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
from pathlib import Path
import json
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

class KolamAugmentationEngine:
    """Advanced augmentation engine for Kolam patterns with cultural preservation"""

    def __init__(self, preserve_symmetry: bool = True, preserve_cultural_rules: bool = True):
        self.preserve_symmetry = preserve_symmetry
        self.preserve_cultural_rules = preserve_cultural_rules
        self.augmentation_count = 0

        # Initialize augmentation pipelines
        self.geometric_transforms = self._create_geometric_pipeline()
        self.photometric_transforms = self._create_photometric_pipeline()
        self.noise_transforms = self._create_noise_pipeline()
        self.artistic_transforms = self._create_artistic_pipeline()

    def _create_geometric_pipeline(self) -> A.Compose:
        """Geometric transformations preserving Kolam structure"""
        return A.Compose([
            A.OneOf([
                A.Rotate(limit=(-10, 10), border_mode=cv2.BORDER_REFLECT, p=0.7),
                A.Rotate(limit=(-2, 2), border_mode=cv2.BORDER_REFLECT, p=0.3),
            ], p=0.8),

            A.OneOf([
                A.RandomScale(scale_limit=0.1, p=0.5),
                A.RandomScale(scale_limit=0.05, p=0.5),
            ], p=0.6),

            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=10, alpha_affine=5, p=0.3),
                A.GridDistortion(num_steps=3, distort_limit=0.1, p=0.4),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.02, p=0.3),
            ], p=0.4),

            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, rotate_limit=2, p=0.4),
                A.Affine(scale=(0.98, 1.02), translate_percent=(-0.02, 0.02), p=0.3),
            ], p=0.5),
        ])

    def _create_photometric_pipeline(self) -> A.Compose:
        """Photometric transformations"""
        return A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.4),
            ], p=0.8),

            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.4),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
                A.ChannelShuffle(p=0.1),
            ], p=0.5),

            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
                A.Equalize(p=0.2),
                A.RandomGamma(gamma_limit=(90, 110), p=0.3),
            ], p=0.4),
        ])

    def _create_noise_pipeline(self) -> A.Compose:
        """Noise and blur transformations"""
        return A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(5, 15), p=0.4),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.3),
                A.MultiplicativeNoise(multiplier=(0.98, 1.02), p=0.3),
            ], p=0.3),

            A.OneOf([
                A.Blur(blur_limit=2, p=0.3),
                A.GaussianBlur(blur_limit=2, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
            ], p=0.25),

            A.OneOf([
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.3),
                A.UnsharpMask(blur_limit=2, sigma_limit=0.5, alpha=(0.1, 0.3), p=0.2),
            ], p=0.2),
        ])

    def _create_artistic_pipeline(self) -> A.Compose:
        """Artistic style transformations"""
        return A.Compose([
            A.OneOf([
                A.Emboss(alpha=(0.1, 0.3), strength=(0.1, 0.4), p=0.2),
                A.Superpixels(p_replace=0.05, n_segments=50, p=0.1),
                A.Posterize(num_bits=6, p=0.1),
            ], p=0.15),
        ])

    def apply_single_augmentation(self, image: np.ndarray, augmentation_type: str = "mixed") -> np.ndarray:
        """Apply single augmentation to image"""

        if augmentation_type == "geometric":
            transformed = self.geometric_transforms(image=image)["image"]
        elif augmentation_type == "photometric":
            transformed = self.photometric_transforms(image=image)["image"]
        elif augmentation_type == "noise":
            transformed = self.noise_transforms(image=image)["image"]
        elif augmentation_type == "artistic":
            transformed = self.artistic_transforms(image=image)["image"]
        else:  # mixed
            # Randomly choose combination of transforms
            transforms_to_apply = []

            if random.random() < 0.7:
                transforms_to_apply.append(self.geometric_transforms)
            if random.random() < 0.6:
                transforms_to_apply.append(self.photometric_transforms)
            if random.random() < 0.3:
                transforms_to_apply.append(self.noise_transforms)
            if random.random() < 0.15:
                transforms_to_apply.append(self.artistic_transforms)

            transformed = image.copy()
            for transform in transforms_to_apply:
                transformed = transform(image=transformed)["image"]

        # Preserve cultural constraints if enabled
        if self.preserve_cultural_rules:
            transformed = self._apply_cultural_constraints(transformed, image)

        return transformed

    def _apply_cultural_constraints(self, transformed_image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """Apply cultural constraints to preserve Kolam authenticity"""

        # Ensure image doesn't lose essential structural elements
        if self._has_lost_structure(transformed_image, original_image):
            # Blend with original to preserve structure
            alpha = 0.7
            transformed_image = cv2.addWeighted(transformed_image, alpha, original_image, 1-alpha, 0)

        return transformed_image

    def _has_lost_structure(self, transformed: np.ndarray, original: np.ndarray, threshold: float = 0.3) -> bool:
        """Check if transformation has destroyed essential structure"""
        # Simple structure preservation check using edge detection

        # Convert to grayscale
        gray_original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) if len(original.shape) == 3 else original
        gray_transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2GRAY) if len(transformed.shape) == 3 else transformed

        # Edge detection
        edges_original = cv2.Canny(gray_original, 50, 150)
        edges_transformed = cv2.Canny(gray_transformed, 50, 150)

        # Compare edge densities
        edge_density_original = np.sum(edges_original > 0) / edges_original.size
        edge_density_transformed = np.sum(edges_transformed > 0) / edges_transformed.size

        # If edge density drops too much, structure might be lost
        return (edge_density_transformed / max(edge_density_original, 1e-6)) < threshold

    def enforce_symmetry(self, image: np.ndarray, symmetry_type: str = "4fold") -> np.ndarray:
        """Enforce symmetry constraints for cultural authenticity"""

        if symmetry_type == "4fold":
            # Enforce 4-fold rotational symmetry
            h, w = image.shape[:2]
            center = (w // 2, h // 2)

            # Get 4 rotated versions
            rot_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rot_180 = cv2.rotate(image, cv2.ROTATE_180)
            rot_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Average all rotations to enforce symmetry
            symmetric = (image.astype(np.float32) + rot_90.astype(np.float32) + 
                        rot_180.astype(np.float32) + rot_270.astype(np.float32)) / 4.0

            return symmetric.astype(np.uint8)

        elif symmetry_type == "bilateral":
            # Enforce bilateral symmetry
            flipped = cv2.flip(image, 1)  # Horizontal flip
            symmetric = (image.astype(np.float32) + flipped.astype(np.float32)) / 2.0
            return symmetric.astype(np.uint8)

        return image

class KolamDatasetAugmenter:
    """Main class for augmenting Kolam dataset"""

    def __init__(self, data_dir: str = "data", output_dir: str = "augment", 
                 target_count: int = 107000, num_workers: int = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.target_count = target_count
        self.num_workers = num_workers or min(mp.cpu_count(), 8)

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Initialize augmentation engine
        self.augmenter = KolamAugmentationEngine()

        # Get list of input images
        self.input_images = self._get_input_images()
        self.images_per_original = target_count // len(self.input_images)

        print(f"📊 Found {len(self.input_images)} input images")
        print(f"🎯 Target: {target_count} augmented images")
        print(f"📈 Will generate {self.images_per_original} variations per original image")

    def _get_input_images(self) -> List[Path]:
        """Get list of input images matching Rangoli(n) pattern"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = []

        for file_path in self.data_dir.iterdir():
            if file_path.suffix.lower() in image_extensions:
                if file_path.stem.startswith('Rangoli(') and file_path.stem.endswith(')'):
                    images.append(file_path)

        return sorted(images)

    def augment_single_image(self, image_path: Path, start_idx: int) -> Dict:
        """Augment a single image and save variations"""

        try:
            # Load image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract original number from filename
            original_name = image_path.stem

            augmented_count = 0
            failed_count = 0

            # Generate variations
            for i in range(self.images_per_original):
                try:
                    # Apply augmentation
                    augmented = self.augmenter.apply_single_augmentation(image)

                    # Generate output filename
                    output_name = f"kolam_{start_idx + i:06d}.jpg"
                    output_path = self.output_dir / output_name

                    # Save augmented image
                    augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_path), augmented_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    augmented_count += 1

                except Exception as e:
                    failed_count += 1
                    if failed_count < 5:  # Only print first few errors
                        print(f"⚠️ Failed to augment {original_name} variation {i}: {e}")

            return {
                'original': original_name,
                'augmented_count': augmented_count,
                'failed_count': failed_count,
                'success_rate': augmented_count / self.images_per_original
            }

        except Exception as e:
            print(f"❌ Failed to process {image_path}: {e}")
            return {
                'original': image_path.stem,
                'augmented_count': 0,
                'failed_count': self.images_per_original,
                'success_rate': 0.0
            }

    def augment_dataset_parallel(self) -> Dict:
        """Augment entire dataset using parallel processing"""

        print(f"🚀 Starting augmentation with {self.num_workers} workers...")

        # Prepare arguments for parallel processing
        args_list = []
        for i, image_path in enumerate(self.input_images):
            start_idx = i * self.images_per_original
            args_list.append((image_path, start_idx))

        # Process in parallel
        with mp.Pool(self.num_workers) as pool:
            results = list(tqdm(
                pool.starmap(self.augment_single_image, args_list),
                total=len(args_list),
                desc="Augmenting images"
            ))

        # Compile statistics
        total_augmented = sum(r['augmented_count'] for r in results)
        total_failed = sum(r['failed_count'] for r in results)
        avg_success_rate = np.mean([r['success_rate'] for r in results])

        stats = {
            'total_original_images': len(self.input_images),
            'total_augmented_images': total_augmented,
            'total_failed_augmentations': total_failed,
            'average_success_rate': avg_success_rate,
            'target_achieved': total_augmented >= self.target_count * 0.95,
            'results_per_image': results
        }

        # Save statistics
        stats_path = self.output_dir / 'augmentation_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        return stats

    def augment_dataset_sequential(self) -> Dict:
        """Augment dataset sequentially (for debugging)"""

        print(f"🐌 Starting sequential augmentation...")

        results = []
        for i, image_path in enumerate(tqdm(self.input_images, desc="Processing images")):
            start_idx = i * self.images_per_original
            result = self.augment_single_image(image_path, start_idx)
            results.append(result)

        # Compile statistics (same as parallel version)
        total_augmented = sum(r['augmented_count'] for r in results)
        total_failed = sum(r['failed_count'] for r in results)
        avg_success_rate = np.mean([r['success_rate'] for r in results])

        stats = {
            'total_original_images': len(self.input_images),
            'total_augmented_images': total_augmented,
            'total_failed_augmentations': total_failed,
            'average_success_rate': avg_success_rate,
            'target_achieved': total_augmented >= self.target_count * 0.95,
            'results_per_image': results
        }

        # Save statistics
        stats_path = self.output_dir / 'augmentation_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        return stats

# Quick test/demo functions
def create_sample_augmentation():
    """Create a few sample augmentations for testing"""

    from PIL import Image
    import matplotlib.pyplot as plt

    # Create a simple test image
    test_image = np.ones((256, 256, 3), dtype=np.uint8) * 255

    # Draw a simple pattern
    cv2.circle(test_image, (128, 128), 50, (0, 0, 0), 2)
    cv2.rectangle(test_image, (100, 100), (156, 156), (0, 0, 0), 2)

    # Initialize augmenter
    augmenter = KolamAugmentationEngine()

    # Create several augmentations
    augmentations = []
    for i in range(6):
        aug = augmenter.apply_single_augmentation(test_image)
        augmentations.append(aug)

    # Display results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(test_image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')

    for i, aug in enumerate(augmentations):
        row = (i + 1) // 4
        col = (i + 1) % 4
        axes[row, col].imshow(aug)
        axes[row, col].set_title(f"Augmentation {i+1}")
        axes[row, col].axis('off')

    # Hide last subplot
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig("augmentation_samples.png", dpi=150, bbox_inches='tight')
    plt.show()

    return augmentations

