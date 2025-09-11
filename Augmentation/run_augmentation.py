
"""
KolamAI/run_augmentation.py
Main script to run Kolam dataset augmentation
Usage: python run_augmentation.py [options]
"""

import argparse
import sys
from pathlib import Path
import time
from typing import Dict

# Import our modules
from kolam_augmentation import KolamDatasetAugmenter
from augmentation_config import aug_config
from augmentation_utils import ImageFileManager, AugmentationReporter

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Augment Kolam dataset from 107 images to 107,000 images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input-dir', 
        type=str, 
        default='data',
        help='Directory containing original Kolam images (Rangoli(n) format)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='augment', 
        help='Output directory for augmented images'
    )

    parser.add_argument(
        '--target-count',
        type=int,
        default=107000,
        help='Target number of augmented images to generate'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker processes (default: auto-detect)'
    )

    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run augmentation sequentially instead of parallel (for debugging)'
    )

    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up partial results from previous runs'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate input directory without running augmentation'
    )

    parser.add_argument(
        '--create-samples',
        action='store_true',
        help='Create sample visualization after augmentation'
    )

    return parser.parse_args()

def validate_setup(args) -> bool:
    """Validate setup before running augmentation"""

    print("🔍 Validating setup...")

    # Check input directory
    is_valid, valid_images = ImageFileManager.validate_input_directory(args.input_dir)
    if not is_valid:
        print(f"❌ No valid Kolam images found in {args.input_dir}")
        print("   Expected format: Rangoli(1).jpg, Rangoli(2).png, etc.")
        return False

    print(f"✅ Found {len(valid_images)} valid input images")

    # Estimate disk space
    space_needed = ImageFileManager.estimate_disk_space(args.target_count)
    print(f"💾 Estimated disk space needed: {space_needed:.1f} GB")

    # Setup output directory
    if not ImageFileManager.setup_output_directory(args.output_dir):
        print(f"❌ Failed to setup output directory: {args.output_dir}")
        return False

    print(f"✅ Output directory ready: {args.output_dir}")

    # Update config
    aug_config.input_dir = args.input_dir
    aug_config.output_dir = args.output_dir
    aug_config.target_total_images = args.target_count
    aug_config.original_image_count = len(valid_images)

    if args.workers:
        aug_config.num_workers = args.workers

    # Show augmentation plan
    images_per_original = aug_config.get_images_per_original()
    print(f"📊 Augmentation plan:")
    print(f"   • {len(valid_images)} original images")
    print(f"   • {images_per_original} augmentations per original")
    print(f"   • {args.target_count:,} total target images")
    print(f"   • {aug_config.num_workers} worker processes")

    return True

def run_augmentation(args) -> Dict:
    """Run the main augmentation process"""

    print("\n🚀 Starting Kolam dataset augmentation...")
    start_time = time.time()

    # Initialize augmenter
    augmenter = KolamDatasetAugmenter(
        data_dir=args.input_dir,
        output_dir=args.output_dir,
        target_count=args.target_count,
        num_workers=args.workers
    )

    # Clean up if requested
    if args.cleanup:
        print("🧹 Cleaning up partial results...")
        cleaned = ImageFileManager.cleanup_partial_results(args.output_dir, args.target_count)
        if cleaned > 0:
            print(f"   Removed {cleaned} partial files")

    # Run augmentation
    if args.sequential:
        print("🐌 Running sequential augmentation (debugging mode)...")
        stats = augmenter.augment_dataset_sequential()
    else:
        print(f"🚀 Running parallel augmentation with {augmenter.num_workers} workers...")
        stats = augmenter.augment_dataset_parallel()

    # Calculate timing
    end_time = time.time()
    duration = end_time - start_time
    images_per_second = stats['total_augmented_images'] / duration

    stats['processing_time'] = duration
    stats['images_per_second'] = images_per_second

    return stats

def generate_reports(args, stats: Dict):
    """Generate reports and visualizations"""

    print("\n📊 Generating reports...")

    # Initialize reporter
    reporter = AugmentationReporter(args.output_dir)

    # Generate summary report
    summary = reporter.generate_summary_report(stats)
    print("✅ Summary report generated")

    # Create sample visualization if requested
    if args.create_samples:
        print("🎨 Creating sample visualization...")
        reporter.create_sample_visualization(args.input_dir, args.output_dir)

    # Print summary to console
    print("\n" + "="*60)
    print("AUGMENTATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📁 Input images: {stats['total_original_images']}")
    print(f"🎨 Generated images: {stats['total_augmented_images']:,}")
    print(f"⏱️ Processing time: {stats['processing_time']:.1f} seconds")
    print(f"🚄 Speed: {stats['images_per_second']:.1f} images/second")
    print(f"✅ Success rate: {stats['average_success_rate']:.2%}")
    print(f"🎯 Target achieved: {'YES' if stats['target_achieved'] else 'NO'}")
    print("="*60)

def main():
    """Main function"""

    args = parse_arguments()

    print("🕸️ KolamAI Dataset Augmentation")
    print("="*50)

    # Validate setup
    if not validate_setup(args):
        sys.exit(1)

    # If validation-only mode, exit here
    if args.validate_only:
        print("\n✅ Validation completed successfully!")
        sys.exit(0)

    try:
        # Run augmentation
        stats = run_augmentation(args)

        # Generate reports
        generate_reports(args, stats)

        print("\n🎉 All done! Your 107,000 augmented Kolam images are ready!")

    except KeyboardInterrupt:
        print("\n⚠️ Augmentation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Augmentation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

