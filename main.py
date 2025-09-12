"""
KolamAI - Main Application Entry Point
Complete solution for SIH Problem Statement ID25107
"""

import argparse
import sys
import os
from pathlib import Path
import json
import time
from typing import Dict, List

# Import our modules
from kolam_analyzer import KolamAnalyzer, analyze_kolam_batch
from kolam_generator import KolamGenerator, GenerationParams, SymmetryType, generate_kolam_variations
from mathematical_principles import MathematicalAnalyzer, analyze_kolam_mathematics
from Augmentation.run_augmentation import main as run_augmentation

def main():
    """Main application entry point"""
    
    parser = argparse.ArgumentParser(
        description="KolamAI - Advanced Kolam Pattern Analysis & Generation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze Kolam patterns')
    analyze_parser.add_argument('--input', '-i', required=True, help='Input image or directory')
    analyze_parser.add_argument('--output', '-o', default='analysis_results.json', help='Output file')
    analyze_parser.add_argument('--batch', action='store_true', help='Batch analysis mode')
    analyze_parser.add_argument('--mathematical', action='store_true', help='Include mathematical analysis')
    analyze_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate Kolam patterns')
    generate_parser.add_argument('--output', '-o', default='generated_kolam.png', help='Output image file')
    generate_parser.add_argument('--size', type=int, default=512, help='Canvas size')
    generate_parser.add_argument('--grid', type=int, default=16, help='Grid size')
    generate_parser.add_argument('--symmetry', choices=['radial', 'bilateral', 'rotational', 'translational'], 
                                default='radial', help='Symmetry type')
    generate_parser.add_argument('--complexity', type=float, default=0.5, help='Complexity level (0.0-1.0)')
    generate_parser.add_argument('--style', choices=['tamil', 'andhra', 'karnataka', 'kerala'], 
                                default='tamil', help='Cultural style')
    generate_parser.add_argument('--colors', choices=['traditional', 'modern', 'monochrome'], 
                                default='traditional', help='Color scheme')
    generate_parser.add_argument('--variations', type=int, default=1, help='Number of variations to generate')
    
    # Augment command
    augment_parser = subparsers.add_parser('augment', help='Augment dataset')
    augment_parser.add_argument('--input-dir', default='data', help='Input directory')
    augment_parser.add_argument('--output-dir', default='augment', help='Output directory')
    augment_parser.add_argument('--target-count', type=int, default=107000, help='Target number of images')
    augment_parser.add_argument('--workers', type=int, help='Number of worker processes')
    
    # Web app command
    web_parser = subparsers.add_parser('web', help='Launch web application')
    web_parser.add_argument('--port', type=int, default=8501, help='Port number')
    web_parser.add_argument('--host', default='localhost', help='Host address')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    demo_parser.add_argument('--input', default='data', help='Input directory for demo')
    demo_parser.add_argument('--output', default='demo_results', help='Output directory for demo')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    try:
        if args.command == 'analyze':
            run_analysis(args)
        elif args.command == 'generate':
            run_generation(args)
        elif args.command == 'augment':
            run_augmentation_wrapper(args)
        elif args.command == 'web':
            run_web_app(args)
        elif args.command == 'demo':
            run_demo(args)
    
    except KeyboardInterrupt:
        print("\n⚠️ Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.command == 'analyze' and hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def run_analysis(args):
    """Run pattern analysis"""
    
    print("🔍 KolamAI Pattern Analysis")
    print("=" * 40)
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    start_time = time.time()
    
    if args.batch or input_path.is_dir():
        # Batch analysis
        print(f"📁 Running batch analysis on directory: {input_path}")
        
        results = analyze_kolam_batch(str(input_path), args.output)
        
        print(f"✅ Batch analysis complete!")
        print(f"📊 Analyzed {len(results)} images")
        print(f"💾 Results saved to: {args.output}")
        
    else:
        # Single image analysis
        print(f"🖼️ Analyzing single image: {input_path}")
        
        analyzer = KolamAnalyzer()
        pattern = analyzer.analyze_image(str(input_path))
        
        # Display results
        print(f"\n📋 Analysis Results for {input_path.name}:")
        print(f"  🔄 Symmetry: {pattern.symmetry_type} (score: {pattern.symmetry_score:.3f})")
        print(f"  🏛️ Cultural: {pattern.cultural_classification}")
        print(f"  📊 Complexity: {pattern.complexity_score:.3f}")
        print(f"  ⚫ Grid: {pattern.grid_structure.get('grid_type', 'none')}")
        print(f"  📐 Elements: {len(pattern.geometric_elements)}")
        
        # Mathematical analysis if requested
        if args.mathematical:
            print(f"\n🧮 Running mathematical analysis...")
            
            math_analyzer = MathematicalAnalyzer()
            principles = math_analyzer.analyze_mathematical_principles(
                cv2.imread(str(input_path))
            )
            
            print(f"📐 Found {len(principles)} mathematical principles:")
            for principle in principles:
                print(f"  • {principle.name} (confidence: {principle.confidence:.3f})")
        
        # Save results
        result_data = {
            'filename': input_path.name,
            'symmetry_type': pattern.symmetry_type,
            'symmetry_score': pattern.symmetry_score,
            'cultural_classification': pattern.cultural_classification,
            'complexity_score': pattern.complexity_score,
            'grid_structure': pattern.grid_structure,
            'geometric_elements': pattern.geometric_elements,
            'mathematical_properties': pattern.mathematical_properties
        }
        
        if args.mathematical:
            result_data['mathematical_principles'] = [
                {
                    'name': p.name,
                    'description': p.description,
                    'formula': p.formula,
                    'parameters': p.parameters,
                    'confidence': p.confidence
                }
                for p in principles
            ]
        
        with open(args.output, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {args.output}")
    
    duration = time.time() - start_time
    print(f"⏱️ Analysis completed in {duration:.2f} seconds")

def run_generation(args):
    """Run pattern generation"""
    
    print("🎨 KolamAI Pattern Generation")
    print("=" * 40)
    
    # Create generation parameters
    params = GenerationParams(
        canvas_size=(args.size, args.size),
        grid_size=args.grid,
        symmetry_type=SymmetryType(args.symmetry),
        complexity_level=args.complexity,
        color_scheme=args.colors,
        cultural_style=args.style
    )
    
    print(f"🎯 Generation Parameters:")
    print(f"  📐 Canvas: {args.size}x{args.size}")
    print(f"  ⚫ Grid: {args.grid}x{args.grid}")
    print(f"  🔄 Symmetry: {args.symmetry}")
    print(f"  📊 Complexity: {args.complexity}")
    print(f"  🏛️ Style: {args.style}")
    print(f"  🎨 Colors: {args.colors}")
    
    start_time = time.time()
    
    if args.variations > 1:
        print(f"\n🎲 Generating {args.variations} variations...")
        
        variations = generate_kolam_variations(params, args.variations)
        
        # Save variations
        output_path = Path(args.output)
        output_dir = output_path.parent
        output_stem = output_path.stem
        output_ext = output_path.suffix
        
        for i, variation in enumerate(variations):
            filename = f"{output_stem}_variation_{i+1}{output_ext}"
            filepath = output_dir / filename
            
            import cv2
            cv2.imwrite(str(filepath), variation)
            print(f"  💾 Saved: {filepath}")
        
    else:
        print(f"\n🎨 Generating single pattern...")
        
        generator = KolamGenerator()
        kolam = generator.generate_kolam(params)
        
        # Save generated pattern
        import cv2
        cv2.imwrite(args.output, kolam)
        print(f"💾 Generated pattern saved to: {args.output}")
    
    duration = time.time() - start_time
    print(f"⏱️ Generation completed in {duration:.2f} seconds")
    print(f"✅ Success! Your Kolam pattern{'s' if args.variations > 1 else ''} {'are' if args.variations > 1 else 'is'} ready!")

def run_augmentation_wrapper(args):
    """Run data augmentation"""
    
    print("🚀 KolamAI Data Augmentation")
    print("=" * 40)
    
    # Prepare arguments for augmentation script
    aug_args = [
        '--input-dir', args.input_dir,
        '--output-dir', args.output_dir,
        '--target-count', str(args.target_count)
    ]
    
    if args.workers:
        aug_args.extend(['--workers', str(args.workers)])
    
    # Import and run augmentation
    import sys
    original_argv = sys.argv
    sys.argv = ['run_augmentation.py'] + aug_args
    
    try:
        run_augmentation()
    finally:
        sys.argv = original_argv

def run_web_app(args):
    """Launch web application"""
    
    print("🌐 Launching KolamAI Web Application")
    print("=" * 40)
    print(f"🔗 URL: http://{args.host}:{args.port}")
    print("⚠️ Press Ctrl+C to stop the server")
    
    import subprocess
    import sys
    
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
        '--server.port', str(args.port),
        '--server.address', args.host
    ]
    
    subprocess.run(cmd)

def run_demo(args):
    """Run comprehensive demonstration"""
    
    print("🎪 KolamAI Comprehensive Demo")
    print("=" * 40)
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Demo steps
    print("\n📋 Demo Plan:")
    print("  1. Analyze sample Kolam patterns")
    print("  2. Extract mathematical principles")
    print("  3. Generate new patterns based on analysis")
    print("  4. Create pattern variations")
    print("  5. Generate comprehensive report")
    
    start_time = time.time()
    
    # Step 1: Analyze patterns
    print("\n🔍 Step 1: Analyzing sample patterns...")
    
    if input_dir.exists():
        sample_images = list(input_dir.glob("*.jpg"))[:5]  # Analyze first 5 images
        
        analysis_results = []
        for img_path in sample_images:
            try:
                analyzer = KolamAnalyzer()
                pattern = analyzer.analyze_image(str(img_path))
                
                analysis_results.append({
                    'filename': img_path.name,
                    'pattern': pattern
                })
                
                print(f"  ✅ Analyzed: {img_path.name}")
                
            except Exception as e:
                print(f"  ❌ Failed: {img_path.name} - {e}")
        
        print(f"📊 Analyzed {len(analysis_results)} patterns successfully")
    
    else:
        print("⚠️ Input directory not found, skipping analysis")
        analysis_results = []
    
    # Step 2: Mathematical analysis
    print("\n🧮 Step 2: Extracting mathematical principles...")
    
    math_principles = []
    if analysis_results:
        sample_image = sample_images[0]  # Use first image for math analysis
        
        try:
            principles = analyze_kolam_mathematics(str(sample_image))
            math_principles = principles
            
            print(f"📐 Found {len(principles)} mathematical principles:")
            for principle in principles[:3]:  # Show first 3
                print(f"  • {principle.name} (confidence: {principle.confidence:.3f})")
        
        except Exception as e:
            print(f"❌ Mathematical analysis failed: {e}")
    
    # Step 3: Generate new patterns
    print("\n🎨 Step 3: Generating new patterns...")
    
    generation_params = [
        GenerationParams(symmetry_type=SymmetryType.RADIAL, cultural_style="tamil", complexity_level=0.7),
        GenerationParams(symmetry_type=SymmetryType.BILATERAL, cultural_style="andhra", complexity_level=0.5),
        GenerationParams(symmetry_type=SymmetryType.ROTATIONAL, cultural_style="karnataka", complexity_level=0.8),
    ]
    
    generator = KolamGenerator()
    generated_patterns = []
    
    for i, params in enumerate(generation_params):
        try:
            kolam = generator.generate_kolam(params)
            
            output_path = output_dir / f"generated_pattern_{i+1}.png"
            import cv2
            cv2.imwrite(str(output_path), kolam)
            
            generated_patterns.append(output_path)
            print(f"  ✅ Generated: {output_path.name}")
        
        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
    
    # Step 4: Create variations
    print("\n🎲 Step 4: Creating pattern variations...")
    
    if generated_patterns:
        try:
            base_params = generation_params[0]  # Use first params as base
            variations = generate_kolam_variations(base_params, 3)
            
            for i, variation in enumerate(variations):
                output_path = output_dir / f"variation_{i+1}.png"
                import cv2
                cv2.imwrite(str(output_path), variation)
                print(f"  ✅ Created: {output_path.name}")
        
        except Exception as e:
            print(f"  ❌ Variation generation failed: {e}")
    
    # Step 5: Generate report
    print("\n📄 Step 5: Generating comprehensive report...")
    
    report_data = {
        'demo_info': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input_directory': str(input_dir),
            'output_directory': str(output_dir),
            'duration_seconds': time.time() - start_time
        },
        'analysis_results': [
            {
                'filename': result['filename'],
                'symmetry_type': result['pattern'].symmetry_type,
                'symmetry_score': result['pattern'].symmetry_score,
                'cultural_classification': result['pattern'].cultural_classification,
                'complexity_score': result['pattern'].complexity_score
            }
            for result in analysis_results
        ],
        'mathematical_principles': [
            {
                'name': p.name,
                'description': p.description,
                'confidence': p.confidence
            }
            for p in math_principles
        ],
        'generated_patterns': [str(p) for p in generated_patterns],
        'statistics': {
            'patterns_analyzed': len(analysis_results),
            'principles_found': len(math_principles),
            'patterns_generated': len(generated_patterns)
        }
    }
    
    report_path = output_dir / 'demo_report.json'
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    # Generate summary
    duration = time.time() - start_time
    
    print(f"\n🎉 Demo Complete!")
    print("=" * 40)
    print(f"⏱️ Total time: {duration:.2f} seconds")
    print(f"📊 Patterns analyzed: {len(analysis_results)}")
    print(f"🧮 Mathematical principles: {len(math_principles)}")
    print(f"🎨 Patterns generated: {len(generated_patterns)}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Report saved: {report_path}")
    
    print(f"\n🏆 KolamAI Demo showcases:")
    print("  ✅ Advanced pattern analysis with symmetry detection")
    print("  ✅ Mathematical principle extraction")
    print("  ✅ Cultural classification (Tamil, Andhra, Karnataka, Kerala)")
    print("  ✅ Intelligent pattern generation")
    print("  ✅ Variation creation with controlled parameters")
    print("  ✅ Comprehensive reporting and documentation")

if __name__ == "__main__":
    main()