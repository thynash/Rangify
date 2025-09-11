
"""
KolamAI/data/augmentation_utils.py
Utility functions for Kolam augmentation
"""

import cv2
import numpy as np
from PIL import Image, ImageStat
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import matplotlib.pyplot as plt
from datetime import datetime

class AugmentationValidator:
    """Validate augmented images for quality and cultural authenticity"""

    def __init__(self):
        self.validation_metrics = {}

    def validate_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Validate basic image quality metrics"""

        # Convert to PIL for easier analysis
        pil_image = Image.fromarray(image)

        # Calculate image statistics
        stat = ImageStat.Stat(pil_image)

        metrics = {
            'brightness': np.mean(stat.mean),
            'contrast': np.mean(stat.stddev),
            'sharpness': self._calculate_sharpness(image),
            'color_diversity': self._calculate_color_diversity(image),
            'noise_level': self._calculate_noise_level(image)
        }

        return metrics

    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _calculate_color_diversity(self, image: np.ndarray) -> float:
        """Calculate color diversity in the image"""
        if len(image.shape) == 3:
            # Calculate histogram for each channel
            hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])

            # Calculate entropy as measure of diversity
            entropy = 0
            for hist in [hist_r, hist_g, hist_b]:
                hist = hist.flatten()
                hist = hist / (hist.sum() + 1e-7)
                entropy += -np.sum(hist * np.log2(hist + 1e-7))

            return entropy / 3.0
        else:
            # Grayscale image
            hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-7)
            return -np.sum(hist * np.log2(hist + 1e-7))

    def _calculate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level in image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Use Laplacian to estimate noise
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.std()

    def validate_cultural_authenticity(self, image: np.ndarray, original: np.ndarray) -> Dict[str, float]:
        """Validate cultural authenticity against original"""

        # Structural similarity
        structural_sim = self._calculate_structural_similarity(image, original)

        # Edge preservation
        edge_preservation = self._calculate_edge_preservation(image, original)

        # Symmetry preservation
        symmetry_score = self._calculate_symmetry_score(image)

        return {
            'structural_similarity': structural_sim,
            'edge_preservation': edge_preservation,
            'symmetry_score': symmetry_score,
            'overall_authenticity': (structural_sim + edge_preservation + symmetry_score) / 3.0
        }

    def _calculate_structural_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate structural similarity between images"""
        # Resize images to same size if needed
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Convert to grayscale
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            gray1, gray2 = img1, img2

        # Simple correlation coefficient
        correlation = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0, 0]
        return max(0.0, correlation)

    def _calculate_edge_preservation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate how well edges are preserved"""
        # Resize if needed
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Convert to grayscale and detect edges
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            gray1, gray2 = img1, img2

        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)

        # Calculate overlap of edges
        overlap = np.logical_and(edges1 > 0, edges2 > 0).sum()
        total_edges = np.logical_or(edges1 > 0, edges2 > 0).sum()

        return overlap / max(total_edges, 1)

    def _calculate_symmetry_score(self, image: np.ndarray) -> float:
        """Calculate symmetry score for the image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Check horizontal symmetry
        flipped = cv2.flip(gray, 1)
        h_symmetry = cv2.matchTemplate(gray, flipped, cv2.TM_CCOEFF_NORMED)[0, 0]

        # Check vertical symmetry
        flipped = cv2.flip(gray, 0)
        v_symmetry = cv2.matchTemplate(gray, flipped, cv2.TM_CCOEFF_NORMED)[0, 0]

        return max(0.0, (h_symmetry + v_symmetry) / 2.0)

class AugmentationReporter:
    """Generate reports and visualizations for augmentation results"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.report_dir = self.output_dir / "reports"
        self.report_dir.mkdir(exist_ok=True)

    def generate_summary_report(self, stats: Dict) -> str:
        """Generate a comprehensive summary report"""

        report_lines = [
            "=" * 60,
            "KOLAM DATASET AUGMENTATION REPORT",
            "=" * 60,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SUMMARY STATISTICS:",
            f"  • Original images processed: {stats['total_original_images']:,}",
            f"  • Total augmented images: {stats['total_augmented_images']:,}",
            f"  • Failed augmentations: {stats['total_failed_augmentations']:,}",
            f"  • Overall success rate: {stats['average_success_rate']:.2%}",
            f"  • Target achieved: {'✅ YES' if stats['target_achieved'] else '❌ NO'}",
            "",
            "PER-IMAGE BREAKDOWN:",
        ]

        # Add per-image statistics
        for result in stats['results_per_image'][:10]:  # Show first 10
            report_lines.append(
                f"  • {result['original']}: {result['augmented_count']:,} images "
                f"({result['success_rate']:.1%} success)"
            )

        if len(stats['results_per_image']) > 10:
            report_lines.append(f"  ... and {len(stats['results_per_image']) - 10} more images")

        report_lines.extend([
            "",
            "QUALITY METRICS:",
            f"  • Average augmentations per original: {stats['total_augmented_images'] // stats['total_original_images']:,}",
            f"  • Data multiplication factor: {stats['total_augmented_images'] / stats['total_original_images']:.1f}x",
            "",
            "=" * 60
        ])

        report_content = "\n".join(report_lines)

        # Save report
        report_path = self.report_dir / "augmentation_summary.txt"
        with open(report_path, 'w') as f:
            f.write(report_content)

        return report_content

    def create_sample_visualization(self, input_dir: str, output_dir: str, num_samples: int = 5):
        """Create visualization showing original vs augmented samples"""

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Get a few original images
        original_images = list(input_path.glob("Rangoli(*"))[:num_samples]

        if not original_images:
            print("⚠️ No original images found for visualization")
            return

        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        fig.suptitle("Original vs Augmented Kolam Patterns", fontsize=16)

        for i, orig_path in enumerate(original_images):
            # Load original
            orig_img = cv2.imread(str(orig_path))
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

            # Find corresponding augmented images
            base_idx = i * 1000  # Assuming 1000 augmentations per original
            aug_paths = [
                output_path / f"kolam_{base_idx + j:06d}.jpg" 
                for j in [0, 100, 500]  # Sample 3 augmented versions
            ]

            # Plot original
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f"Original: {orig_path.stem}")
            axes[i, 0].axis('off')

            # Plot augmented versions
            for j, aug_path in enumerate(aug_paths):
                if aug_path.exists():
                    aug_img = cv2.imread(str(aug_path))
                    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                    axes[i, j + 1].imshow(aug_img)
                    axes[i, j + 1].set_title(f"Augmented {j + 1}")
                else:
                    axes[i, j + 1].text(0.5, 0.5, "Not Found", ha='center', va='center')
                axes[i, j + 1].axis('off')

        plt.tight_layout()
        viz_path = self.report_dir / "augmentation_samples.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 Visualization saved to {viz_path}")

class ImageFileManager:
    """Manage image file operations for augmentation"""

    @staticmethod
    def validate_input_directory(input_dir: str) -> Tuple[bool, List[str]]:
        """Validate input directory and return list of valid images"""

        input_path = Path(input_dir)
        if not input_path.exists():
            return False, []

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        valid_images = []

        for file_path in input_path.iterdir():
            if (file_path.suffix.lower() in valid_extensions and 
                file_path.stem.startswith('Rangoli(') and 
                file_path.stem.endswith(')')):
                valid_images.append(str(file_path))

        return len(valid_images) > 0, valid_images

    @staticmethod
    def setup_output_directory(output_dir: str) -> bool:
        """Setup output directory structure"""

        output_path = Path(output_dir)
        try:
            output_path.mkdir(exist_ok=True)

            # Create subdirectories
            (output_path / "reports").mkdir(exist_ok=True)
            (output_path / "samples").mkdir(exist_ok=True)

            return True
        except Exception as e:
            print(f"❌ Failed to setup output directory: {e}")
            return False

    @staticmethod
    def estimate_disk_space(num_images: int, avg_image_size_mb: float = 0.5) -> float:
        """Estimate required disk space in GB"""
        return (num_images * avg_image_size_mb) / 1024

    @staticmethod
    def cleanup_partial_results(output_dir: str, expected_count: int) -> int:
        """Clean up partial results from failed runs"""

        output_path = Path(output_dir)
        if not output_path.exists():
            return 0

        # Count existing files
        existing_files = list(output_path.glob("kolam_*.jpg"))

        if len(existing_files) < expected_count:
            print(f"🧹 Found {len(existing_files)} partial results, cleaning up...")

            for file_path in existing_files:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"⚠️ Failed to delete {file_path}: {e}")

            return len(existing_files)

        return 0

