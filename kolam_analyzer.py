"""
KolamAI - Advanced Pattern Analysis Engine
Identifies mathematical principles and design patterns in Kolam/Rangoli images
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure, morphology, feature
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
import networkx as nx
from shapely.geometry import Point, Polygon, LineString
import math

@dataclass
class KolamPattern:
    """Data structure to hold analyzed Kolam pattern information"""
    symmetry_type: str
    symmetry_score: float
    grid_structure: Dict
    geometric_elements: List[Dict]
    mathematical_properties: Dict
    cultural_classification: str
    complexity_score: float
    
class KolamAnalyzer:
    """Advanced analyzer for Kolam patterns and mathematical principles"""
    
    def __init__(self):
        self.pattern_database = {}
        self.symmetry_types = ['radial', 'bilateral', 'rotational', 'translational']
        self.geometric_primitives = ['circle', 'line', 'curve', 'dot', 'loop']
        
    def analyze_image(self, image_path: str) -> KolamPattern:
        """Complete analysis of a Kolam image"""
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Core analysis components
        symmetry_analysis = self._analyze_symmetry(gray)
        grid_analysis = self._analyze_grid_structure(gray)
        geometric_analysis = self._analyze_geometric_elements(gray)
        mathematical_analysis = self._analyze_mathematical_properties(gray)
        cultural_analysis = self._classify_cultural_pattern(image, gray)
        
        # Calculate overall complexity
        complexity = self._calculate_complexity_score(
            symmetry_analysis, grid_analysis, geometric_analysis
        )
        
        return KolamPattern(
            symmetry_type=symmetry_analysis['primary_type'],
            symmetry_score=symmetry_analysis['score'],
            grid_structure=grid_analysis,
            geometric_elements=geometric_analysis,
            mathematical_properties=mathematical_analysis,
            cultural_classification=cultural_analysis,
            complexity_score=complexity
        )
    
    def _analyze_symmetry(self, gray_image: np.ndarray) -> Dict:
        """Analyze symmetry patterns in the Kolam"""
        
        h, w = gray_image.shape
        center_x, center_y = w // 2, h // 2
        
        symmetry_scores = {}
        
        # Bilateral symmetry (vertical axis)
        left_half = gray_image[:, :center_x]
        right_half = cv2.flip(gray_image[:, center_x:], 1)
        if left_half.shape == right_half.shape:
            symmetry_scores['bilateral_vertical'] = cv2.matchTemplate(
                left_half, right_half, cv2.TM_CCOEFF_NORMED
            )[0, 0]
        
        # Bilateral symmetry (horizontal axis)
        top_half = gray_image[:center_y, :]
        bottom_half = cv2.flip(gray_image[center_y:, :], 0)
        if top_half.shape == bottom_half.shape:
            symmetry_scores['bilateral_horizontal'] = cv2.matchTemplate(
                top_half, bottom_half, cv2.TM_CCOEFF_NORMED
            )[0, 0]
        
        # Rotational symmetry (90, 180, 270 degrees)
        for angle in [90, 180, 270]:
            rotated = self._rotate_image(gray_image, angle)
            if rotated.shape == gray_image.shape:
                score = cv2.matchTemplate(
                    gray_image, rotated, cv2.TM_CCOEFF_NORMED
                )[0, 0]
                symmetry_scores[f'rotational_{angle}'] = score
        
        # Radial symmetry analysis
        radial_score = self._analyze_radial_symmetry(gray_image)
        symmetry_scores['radial'] = radial_score
        
        # Determine primary symmetry type
        primary_type = max(symmetry_scores.items(), key=lambda x: x[1])
        
        return {
            'scores': symmetry_scores,
            'primary_type': primary_type[0],
            'score': primary_type[1],
            'is_symmetric': primary_type[1] > 0.7
        }
    
    def _analyze_radial_symmetry(self, gray_image: np.ndarray) -> float:
        """Analyze radial symmetry by comparing angular segments"""
        
        h, w = gray_image.shape
        center_x, center_y = w // 2, h // 2
        
        # Convert to polar coordinates
        max_radius = min(center_x, center_y)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        
        segments = []
        for i, angle in enumerate(angles):
            # Extract radial line
            x_coords = center_x + np.arange(max_radius) * np.cos(angle)
            y_coords = center_y + np.arange(max_radius) * np.sin(angle)
            
            # Ensure coordinates are within bounds
            valid_mask = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
            x_coords = x_coords[valid_mask].astype(int)
            y_coords = y_coords[valid_mask].astype(int)
            
            if len(x_coords) > 0:
                segment = gray_image[y_coords, x_coords]
                segments.append(segment)
        
        # Calculate similarity between segments
        if len(segments) < 2:
            return 0.0
            
        similarities = []
        for i in range(len(segments)):
            for j in range(i+1, len(segments)):
                # Pad shorter segment
                seg1, seg2 = segments[i], segments[j]
                min_len = min(len(seg1), len(seg2))
                if min_len > 0:
                    corr = np.corrcoef(seg1[:min_len], seg2[:min_len])[0, 1]
                    if not np.isnan(corr):
                        similarities.append(abs(corr))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _analyze_grid_structure(self, gray_image: np.ndarray) -> Dict:
        """Analyze underlying grid structure and dot patterns"""
        
        # Detect potential dot/grid points using blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 500
        params.filterByCircularity = True
        params.minCircularity = 0.3
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(255 - gray_image)  # Invert for dark dots
        
        if len(keypoints) < 4:
            return {'has_grid': False, 'grid_type': 'none', 'dot_count': 0}
        
        # Extract dot positions
        dot_positions = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        
        # Analyze grid pattern
        grid_analysis = self._analyze_dot_grid(dot_positions)
        
        # Detect grid lines using Hough transform
        edges = cv2.Canny(gray_image, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        line_analysis = self._analyze_grid_lines(lines) if lines is not None else {}
        
        return {
            'has_grid': len(keypoints) > 9,
            'dot_count': len(keypoints),
            'dot_positions': dot_positions.tolist(),
            'grid_spacing': grid_analysis.get('spacing', 0),
            'grid_type': grid_analysis.get('type', 'irregular'),
            'line_analysis': line_analysis
        }
    
    def _analyze_dot_grid(self, positions: np.ndarray) -> Dict:
        """Analyze the arrangement of dots to determine grid type"""
        
        if len(positions) < 4:
            return {'type': 'insufficient_data', 'spacing': 0}
        
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(positions)
        
        # Find most common distance (grid spacing)
        hist, bins = np.histogram(distances, bins=50)
        most_common_distance = bins[np.argmax(hist)]
        
        # Cluster similar distances to find grid spacing
        distance_clusters = DBSCAN(eps=most_common_distance*0.2).fit(distances.reshape(-1, 1))
        
        # Analyze arrangement pattern
        if len(np.unique(distance_clusters.labels_)) <= 3:
            grid_type = 'regular'
        else:
            grid_type = 'irregular'
        
        return {
            'type': grid_type,
            'spacing': most_common_distance,
            'distance_variance': np.var(distances)
        }
    
    def _analyze_geometric_elements(self, gray_image: np.ndarray) -> List[Dict]:
        """Identify and classify geometric elements in the pattern"""
        
        elements = []
        
        # Edge detection for line analysis
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                elements.append({
                    'type': 'circle',
                    'center': (int(x), int(y)),
                    'radius': int(r),
                    'area': np.pi * r * r
                })
        
        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180, threshold=50,
            minLineLength=30, maxLineGap=10
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                elements.append({
                    'type': 'line',
                    'start': (int(x1), int(y1)),
                    'end': (int(x2), int(y2)),
                    'length': length,
                    'angle': angle
                })
        
        # Detect contours for complex shapes
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small noise
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Classify based on number of vertices
                vertices = len(approx)
                if vertices == 3:
                    shape_type = 'triangle'
                elif vertices == 4:
                    shape_type = 'quadrilateral'
                elif vertices > 8:
                    shape_type = 'complex_curve'
                else:
                    shape_type = f'{vertices}_sided_polygon'
                
                elements.append({
                    'type': shape_type,
                    'contour': contour.tolist(),
                    'area': cv2.contourArea(contour),
                    'perimeter': cv2.arcLength(contour, True),
                    'vertices': vertices
                })
        
        return elements
    
    def _analyze_mathematical_properties(self, gray_image: np.ndarray) -> Dict:
        """Extract mathematical properties and relationships"""
        
        properties = {}
        
        # Fractal dimension estimation
        properties['fractal_dimension'] = self._estimate_fractal_dimension(gray_image)
        
        # Entropy and complexity measures
        properties['entropy'] = self._calculate_image_entropy(gray_image)
        
        # Frequency domain analysis
        fft = np.fft.fft2(gray_image)
        fft_magnitude = np.abs(fft)
        properties['frequency_peak'] = np.max(fft_magnitude)
        properties['frequency_mean'] = np.mean(fft_magnitude)
        
        # Texture analysis using Local Binary Patterns
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
        properties['texture_uniformity'] = len(np.unique(lbp))
        
        # Geometric ratios (Golden ratio, etc.)
        h, w = gray_image.shape
        properties['aspect_ratio'] = w / h
        properties['golden_ratio_similarity'] = abs((w/h) - 1.618) < 0.1
        
        return properties
    
    def _estimate_fractal_dimension(self, image: np.ndarray) -> float:
        """Estimate fractal dimension using box-counting method"""
        
        # Binarize image
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Box counting
        sizes = [2, 4, 8, 16, 32, 64]
        counts = []
        
        for size in sizes:
            # Divide image into boxes of given size
            h, w = binary.shape
            boxes_h = h // size
            boxes_w = w // size
            
            count = 0
            for i in range(boxes_h):
                for j in range(boxes_w):
                    box = binary[i*size:(i+1)*size, j*size:(j+1)*size]
                    if np.any(box == 0):  # Box contains pattern
                        count += 1
            
            counts.append(count)
        
        # Calculate fractal dimension
        if len(counts) > 1 and all(c > 0 for c in counts):
            log_sizes = np.log(sizes)
            log_counts = np.log(counts)
            
            # Linear regression
            coeffs = np.polyfit(log_sizes, log_counts, 1)
            fractal_dim = -coeffs[0]  # Negative slope
            
            return max(0, min(2, fractal_dim))  # Clamp to reasonable range
        
        return 1.0  # Default value
    
    def _calculate_image_entropy(self, image: np.ndarray) -> float:
        """Calculate Shannon entropy of image"""
        
        # Calculate histogram
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        
        # Normalize to probabilities
        hist = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return entropy
    
    def _classify_cultural_pattern(self, color_image: np.ndarray, gray_image: np.ndarray) -> str:
        """Classify the cultural/regional style of the Kolam"""
        
        # Analyze color usage
        color_analysis = self._analyze_color_palette(color_image)
        
        # Analyze pattern complexity
        complexity_features = {
            'edge_density': np.sum(cv2.Canny(gray_image, 50, 150) > 0) / gray_image.size,
            'color_diversity': color_analysis['unique_colors'],
            'symmetry_complexity': len([s for s in self._analyze_symmetry(gray_image)['scores'].values() if s > 0.5])
        }
        
        # Simple classification based on features
        if complexity_features['edge_density'] > 0.1 and complexity_features['color_diversity'] > 10:
            return 'complex_traditional'
        elif complexity_features['symmetry_complexity'] >= 3:
            return 'geometric_mandala'
        elif color_analysis['is_monochrome']:
            return 'simple_traditional'
        else:
            return 'modern_decorative'
    
    def _analyze_color_palette(self, color_image: np.ndarray) -> Dict:
        """Analyze color usage in the image"""
        
        # Reshape image for clustering
        pixels = color_image.reshape(-1, 3)
        
        # Use KMeans to find dominant colors
        kmeans = KMeans(n_clusters=min(8, len(np.unique(pixels, axis=0))), random_state=42)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        
        # Calculate color statistics
        unique_colors = len(colors)
        is_monochrome = unique_colors <= 2
        
        # Analyze color harmony
        hsv_colors = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0,0] for color in colors]
        hue_variance = np.var([hsv[0] for hsv in hsv_colors])
        
        return {
            'dominant_colors': colors.tolist(),
            'unique_colors': unique_colors,
            'is_monochrome': is_monochrome,
            'hue_variance': float(hue_variance),
            'color_harmony': 'monochromatic' if hue_variance < 10 else 'diverse'
        }
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        return rotated
    
    def _analyze_grid_lines(self, lines: np.ndarray) -> Dict:
        """Analyze detected grid lines"""
        
        if lines is None or len(lines) == 0:
            return {'line_count': 0}
        
        # Extract line parameters
        rhos = lines[:, 0, 0]
        thetas = lines[:, 0, 1]
        
        # Classify lines by angle
        horizontal_lines = np.sum(np.abs(thetas - np.pi/2) < 0.1)
        vertical_lines = np.sum(np.abs(thetas) < 0.1)
        diagonal_lines = len(lines) - horizontal_lines - vertical_lines
        
        return {
            'line_count': len(lines),
            'horizontal_lines': int(horizontal_lines),
            'vertical_lines': int(vertical_lines),
            'diagonal_lines': int(diagonal_lines),
            'average_spacing': float(np.mean(np.diff(np.sort(rhos))))
        }
    
    def _calculate_complexity_score(self, symmetry_analysis: Dict, 
                                  grid_analysis: Dict, 
                                  geometric_analysis: List[Dict]) -> float:
        """Calculate overall complexity score for the pattern"""
        
        # Symmetry complexity (more symmetry types = higher complexity)
        symmetry_score = len([s for s in symmetry_analysis['scores'].values() if s > 0.5]) / 4.0
        
        # Grid complexity
        grid_score = min(grid_analysis.get('dot_count', 0) / 50.0, 1.0)
        
        # Geometric element complexity
        element_score = min(len(geometric_analysis) / 20.0, 1.0)
        
        # Weighted average
        complexity = (symmetry_score * 0.3 + grid_score * 0.3 + element_score * 0.4)
        
        return min(complexity, 1.0)

def analyze_kolam_batch(image_directory: str, output_file: str = "kolam_analysis.json"):
    """Analyze a batch of Kolam images and save results"""
    
    import os
    from pathlib import Path
    
    analyzer = KolamAnalyzer()
    results = {}
    
    image_dir = Path(image_directory)
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    print(f"Analyzing {len(image_files)} Kolam images...")
    
    for i, image_path in enumerate(image_files):
        try:
            print(f"Processing {image_path.name} ({i+1}/{len(image_files)})")
            
            pattern = analyzer.analyze_image(str(image_path))
            
            # Convert to serializable format
            results[image_path.name] = {
                'symmetry_type': pattern.symmetry_type,
                'symmetry_score': pattern.symmetry_score,
                'grid_structure': pattern.grid_structure,
                'geometric_elements': pattern.geometric_elements,
                'mathematical_properties': pattern.mathematical_properties,
                'cultural_classification': pattern.cultural_classification,
                'complexity_score': pattern.complexity_score
            }
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            results[image_path.name] = {'error': str(e)}
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Analysis complete! Results saved to {output_file}")
    return results

if __name__ == "__main__":
    # Example usage
    analyzer = KolamAnalyzer()
    
    # Analyze single image
    try:
        pattern = analyzer.analyze_image("data/Rangoli (1).jpg")
        print(f"Pattern Analysis:")
        print(f"  Symmetry: {pattern.symmetry_type} (score: {pattern.symmetry_score:.3f})")
        print(f"  Grid: {pattern.grid_structure.get('grid_type', 'none')}")
        print(f"  Elements: {len(pattern.geometric_elements)}")
        print(f"  Cultural: {pattern.cultural_classification}")
        print(f"  Complexity: {pattern.complexity_score:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure you have Kolam images in the data directory")