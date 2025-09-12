"""
KolamAI - Mathematical Principles Engine
Deep mathematical analysis of Kolam patterns and design principles
"""

import numpy as np
import cv2
from scipy import ndimage, spatial, optimize
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import networkx as nx
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union

@dataclass
class MathematicalPrinciple:
    """Data structure for mathematical principles"""
    name: str
    description: str
    formula: str
    parameters: Dict
    confidence: float
    visual_evidence: Optional[np.ndarray] = None

class MathematicalAnalyzer:
    """Advanced mathematical analysis of Kolam patterns"""
    
    def __init__(self):
        self.principles = []
        self.geometric_constants = {
            'golden_ratio': 1.618033988749,
            'pi': np.pi,
            'sqrt_2': np.sqrt(2),
            'sqrt_3': np.sqrt(3),
            'euler': np.e
        }
    
    def analyze_mathematical_principles(self, image: np.ndarray) -> List[MathematicalPrinciple]:
        """Comprehensive mathematical analysis of Kolam pattern"""
        
        principles = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 1. Geometric Proportion Analysis
        principles.extend(self._analyze_geometric_proportions(gray))
        
        # 2. Symmetry Group Analysis
        principles.extend(self._analyze_symmetry_groups(gray))
        
        # 3. Fractal Analysis
        principles.extend(self._analyze_fractal_properties(gray))
        
        # 4. Topology Analysis
        principles.extend(self._analyze_topological_properties(gray))
        
        # 5. Harmonic Analysis
        principles.extend(self._analyze_harmonic_properties(gray))
        
        # 6. Graph Theory Analysis
        principles.extend(self._analyze_graph_properties(gray))
        
        # 7. Information Theory Analysis
        principles.extend(self._analyze_information_properties(gray))
        
        return principles
    
    def _analyze_geometric_proportions(self, image: np.ndarray) -> List[MathematicalPrinciple]:
        """Analyze geometric proportions and ratios"""
        
        principles = []
        h, w = image.shape
        
        # Golden ratio analysis
        aspect_ratio = w / h
        golden_ratio_error = abs(aspect_ratio - self.geometric_constants['golden_ratio'])
        
        if golden_ratio_error < 0.1:
            principles.append(MathematicalPrinciple(
                name="Golden Ratio Proportion",
                description="Pattern dimensions follow the golden ratio φ = 1.618...",
                formula="φ = (1 + √5) / 2",
                parameters={'ratio': aspect_ratio, 'error': golden_ratio_error},
                confidence=1.0 - golden_ratio_error
            ))
        
        # Detect geometric shapes and analyze their proportions
        contours, _ = cv2.findContours(
            cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1],
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Analyze circle-to-square ratios
        circles, squares = self._classify_shapes(contours)
        
        if circles and squares:
            circle_areas = [cv2.contourArea(c) for c in circles]
            square_areas = [cv2.contourArea(s) for s in squares]
            
            if circle_areas and square_areas:
                avg_circle_area = np.mean(circle_areas)
                avg_square_area = np.mean(square_areas)
                ratio = avg_circle_area / avg_square_area
                
                # π/4 is the theoretical ratio for inscribed circle in square
                theoretical_ratio = np.pi / 4
                error = abs(ratio - theoretical_ratio)
                
                if error < 0.2:
                    principles.append(MathematicalPrinciple(
                        name="Circle-Square Proportion",
                        description="Circles and squares follow π/4 area ratio",
                        formula="A_circle / A_square = π/4",
                        parameters={'ratio': ratio, 'theoretical': theoretical_ratio, 'error': error},
                        confidence=1.0 - error
                    ))
        
        # Analyze radial proportions
        radial_props = self._analyze_radial_proportions(image)
        if radial_props:
            principles.append(radial_props)
        
        return principles
    
    def _analyze_symmetry_groups(self, image: np.ndarray) -> List[MathematicalPrinciple]:
        """Analyze symmetry groups and crystallographic properties"""
        
        principles = []
        
        # Detect point group symmetries
        point_group = self._detect_point_group(image)
        
        if point_group:
            principles.append(MathematicalPrinciple(
                name=f"Point Group Symmetry: {point_group['name']}",
                description=f"Pattern exhibits {point_group['name']} point group symmetry",
                formula=point_group['notation'],
                parameters=point_group['parameters'],
                confidence=point_group['confidence']
            ))
        
        # Analyze wallpaper group (plane symmetry)
        wallpaper_group = self._detect_wallpaper_group(image)
        
        if wallpaper_group:
            principles.append(MathematicalPrinciple(
                name=f"Wallpaper Group: {wallpaper_group['name']}",
                description=f"Pattern follows {wallpaper_group['name']} plane symmetry",
                formula=wallpaper_group['notation'],
                parameters=wallpaper_group['parameters'],
                confidence=wallpaper_group['confidence']
            ))
        
        return principles
    
    def _analyze_fractal_properties(self, image: np.ndarray) -> List[MathematicalPrinciple]:
        """Analyze fractal dimensions and self-similarity"""
        
        principles = []
        
        # Box-counting fractal dimension
        fractal_dim = self._calculate_box_counting_dimension(image)
        
        if fractal_dim > 1.1:  # Non-trivial fractal dimension
            principles.append(MathematicalPrinciple(
                name="Fractal Dimension",
                description=f"Pattern exhibits fractal properties with dimension {fractal_dim:.3f}",
                formula="D = lim(ε→0) log(N(ε)) / log(1/ε)",
                parameters={'dimension': fractal_dim, 'method': 'box_counting'},
                confidence=min(1.0, (fractal_dim - 1.0) * 2)
            ))
        
        # Self-similarity analysis
        self_similarity = self._analyze_self_similarity(image)
        
        if self_similarity['score'] > 0.7:
            principles.append(MathematicalPrinciple(
                name="Self-Similarity",
                description="Pattern exhibits self-similar structure at multiple scales",
                formula="S(r) = correlation(P, P_scaled(r))",
                parameters=self_similarity,
                confidence=self_similarity['score']
            ))
        
        return principles
    
    def _analyze_topological_properties(self, image: np.ndarray) -> List[MathematicalPrinciple]:
        """Analyze topological properties like genus, connectivity"""
        
        principles = []
        
        # Calculate Euler characteristic
        euler_char = self._calculate_euler_characteristic(image)
        
        if euler_char != 1:  # Non-trivial topology
            principles.append(MathematicalPrinciple(
                name="Euler Characteristic",
                description=f"Pattern has Euler characteristic χ = {euler_char}",
                formula="χ = V - E + F",
                parameters={'euler_characteristic': euler_char},
                confidence=0.8
            ))
        
        # Analyze connectivity and holes
        connectivity = self._analyze_connectivity(image)
        
        if connectivity['holes'] > 0:
            principles.append(MathematicalPrinciple(
                name="Topological Holes",
                description=f"Pattern contains {connectivity['holes']} topological holes",
                formula="Genus g = (2 - χ) / 2",
                parameters=connectivity,
                confidence=0.9
            ))
        
        return principles
    
    def _analyze_harmonic_properties(self, image: np.ndarray) -> List[MathematicalPrinciple]:
        """Analyze harmonic and frequency domain properties"""
        
        principles = []
        
        # Fourier analysis
        fft = np.fft.fft2(image)
        fft_magnitude = np.abs(fft)
        
        # Find dominant frequencies
        dominant_freqs = self._find_dominant_frequencies(fft_magnitude)
        
        if dominant_freqs:
            # Check for harmonic relationships
            harmonic_ratios = self._analyze_harmonic_ratios(dominant_freqs)
            
            if harmonic_ratios['is_harmonic']:
                principles.append(MathematicalPrinciple(
                    name="Harmonic Frequency Structure",
                    description="Pattern frequencies follow harmonic relationships",
                    formula="f_n = n × f_0",
                    parameters=harmonic_ratios,
                    confidence=harmonic_ratios['confidence']
                ))
        
        # Analyze radial frequency distribution
        radial_spectrum = self._calculate_radial_spectrum(fft_magnitude)
        
        # Check for power law distribution
        power_law = self._fit_power_law(radial_spectrum)
        
        if power_law['r_squared'] > 0.8:
            principles.append(MathematicalPrinciple(
                name="Power Law Frequency Distribution",
                description="Radial frequency spectrum follows power law",
                formula="P(f) = A × f^(-α)",
                parameters=power_law,
                confidence=power_law['r_squared']
            ))
        
        return principles
    
    def _analyze_graph_properties(self, image: np.ndarray) -> List[MathematicalPrinciple]:
        """Analyze graph-theoretic properties of the pattern"""
        
        principles = []
        
        # Convert pattern to graph representation
        graph = self._image_to_graph(image)
        
        if graph and len(graph.nodes()) > 3:
            # Calculate graph properties
            properties = {
                'nodes': len(graph.nodes()),
                'edges': len(graph.edges()),
                'density': nx.density(graph),
                'clustering': nx.average_clustering(graph),
                'diameter': nx.diameter(graph) if nx.is_connected(graph) else None
            }
            
            # Check for small-world properties
            if properties['clustering'] > 0.3 and properties['diameter'] and properties['diameter'] < np.log(properties['nodes']):
                principles.append(MathematicalPrinciple(
                    name="Small-World Network",
                    description="Pattern structure exhibits small-world network properties",
                    formula="C >> C_random and L ≈ L_random",
                    parameters=properties,
                    confidence=0.7
                ))
            
            # Check for scale-free properties
            degree_sequence = [d for n, d in graph.degree()]
            power_law_fit = self._fit_power_law(degree_sequence)
            
            if power_law_fit['r_squared'] > 0.8:
                principles.append(MathematicalPrinciple(
                    name="Scale-Free Network",
                    description="Pattern connectivity follows scale-free distribution",
                    formula="P(k) ∝ k^(-γ)",
                    parameters=power_law_fit,
                    confidence=power_law_fit['r_squared']
                ))
        
        return principles
    
    def _analyze_information_properties(self, image: np.ndarray) -> List[MathematicalPrinciple]:
        """Analyze information-theoretic properties"""
        
        principles = []
        
        # Calculate Shannon entropy
        entropy = self._calculate_shannon_entropy(image)
        
        # Calculate complexity measures
        lz_complexity = self._calculate_lz_complexity(image)
        
        # Analyze information distribution
        if entropy > 6.0:  # High entropy
            principles.append(MathematicalPrinciple(
                name="High Information Content",
                description="Pattern has high Shannon entropy indicating complex information structure",
                formula="H = -Σ p(x) log₂ p(x)",
                parameters={'entropy': entropy, 'max_entropy': 8.0},
                confidence=min(1.0, entropy / 8.0)
            ))
        
        # Check for optimal information distribution
        if 0.7 < lz_complexity < 0.9:
            principles.append(MathematicalPrinciple(
                name="Optimal Complexity",
                description="Pattern exhibits optimal balance between order and randomness",
                formula="C_LZ = length(compressed) / length(original)",
                parameters={'lz_complexity': lz_complexity},
                confidence=0.8
            ))
        
        return principles
    
    # Helper methods for mathematical analysis
    
    def _classify_shapes(self, contours: List) -> Tuple[List, List]:
        """Classify contours into circles and squares"""
        
        circles, squares = [], []
        
        for contour in contours:
            if cv2.contourArea(contour) < 100:  # Skip small contours
                continue
            
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Classify based on number of vertices and circularity
            if len(approx) > 8:  # Many vertices - likely circle
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.7:
                    circles.append(contour)
            elif len(approx) == 4:  # Four vertices - likely square
                squares.append(contour)
        
        return circles, squares
    
    def _analyze_radial_proportions(self, image: np.ndarray) -> Optional[MathematicalPrinciple]:
        """Analyze radial proportions from center"""
        
        h, w = image.shape
        center = (w // 2, h // 2)
        
        # Sample radial lines
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radial_profiles = []
        
        for angle in angles:
            # Extract radial line
            max_radius = min(center[0], center[1], w - center[0], h - center[1])
            x_coords = center[0] + np.arange(max_radius) * np.cos(angle)
            y_coords = center[1] + np.arange(max_radius) * np.sin(angle)
            
            # Ensure coordinates are within bounds
            valid_mask = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
            x_coords = x_coords[valid_mask].astype(int)
            y_coords = y_coords[valid_mask].astype(int)
            
            if len(x_coords) > 0:
                profile = image[y_coords, x_coords]
                radial_profiles.append(profile)
        
        if not radial_profiles:
            return None
        
        # Analyze for geometric progressions
        # Find peaks in radial profiles
        peak_positions = []
        for profile in radial_profiles:
            peaks, _ = self._find_peaks(profile)
            if len(peaks) > 1:
                peak_positions.append(peaks)
        
        if peak_positions:
            # Check for geometric progression in peak spacing
            avg_peaks = np.mean([np.diff(peaks) for peaks in peak_positions if len(peaks) > 1], axis=0)
            
            if len(avg_peaks) > 1:
                ratios = avg_peaks[1:] / avg_peaks[:-1]
                ratio_std = np.std(ratios)
                
                if ratio_std < 0.2:  # Consistent ratio
                    avg_ratio = np.mean(ratios)
                    
                    return MathematicalPrinciple(
                        name="Geometric Radial Progression",
                        description=f"Radial elements follow geometric progression with ratio {avg_ratio:.3f}",
                        formula="r_n = r_0 × q^n",
                        parameters={'ratio': avg_ratio, 'consistency': 1.0 - ratio_std},
                        confidence=1.0 - ratio_std
                    )
        
        return None
    
    def _detect_point_group(self, image: np.ndarray) -> Optional[Dict]:
        """Detect crystallographic point group"""
        
        # Test for various point group symmetries
        symmetries = {}
        
        # Test rotational symmetries
        for n in [2, 3, 4, 6, 8]:
            angle = 2 * np.pi / n
            rotated = self._rotate_image(image, angle * 180 / np.pi)
            
            if rotated.shape == image.shape:
                similarity = cv2.matchTemplate(image, rotated, cv2.TM_CCOEFF_NORMED)[0, 0]
                symmetries[f'C{n}'] = similarity
        
        # Test mirror symmetries
        h_mirror = cv2.flip(image, 1)
        v_mirror = cv2.flip(image, 0)
        
        symmetries['mirror_h'] = cv2.matchTemplate(image, h_mirror, cv2.TM_CCOEFF_NORMED)[0, 0]
        symmetries['mirror_v'] = cv2.matchTemplate(image, v_mirror, cv2.TM_CCOEFF_NORMED)[0, 0]
        
        # Determine point group
        max_symmetry = max(symmetries.items(), key=lambda x: x[1])
        
        if max_symmetry[1] > 0.8:
            return {
                'name': max_symmetry[0],
                'notation': max_symmetry[0],
                'parameters': symmetries,
                'confidence': max_symmetry[1]
            }
        
        return None
    
    def _detect_wallpaper_group(self, image: np.ndarray) -> Optional[Dict]:
        """Detect wallpaper group (plane symmetry group)"""
        
        # Simplified wallpaper group detection
        # In practice, this would be much more complex
        
        # Check for translational symmetry
        h, w = image.shape
        
        # Test horizontal translation
        best_h_translation = 0
        best_h_score = 0
        
        for dx in range(w // 8, w // 2, w // 16):
            translated = np.roll(image, dx, axis=1)
            score = cv2.matchTemplate(image, translated, cv2.TM_CCOEFF_NORMED)[0, 0]
            
            if score > best_h_score:
                best_h_score = score
                best_h_translation = dx
        
        # Test vertical translation
        best_v_translation = 0
        best_v_score = 0
        
        for dy in range(h // 8, h // 2, h // 16):
            translated = np.roll(image, dy, axis=0)
            score = cv2.matchTemplate(image, translated, cv2.TM_CCOEFF_NORMED)[0, 0]
            
            if score > best_v_score:
                best_v_score = score
                best_v_translation = dy
        
        # Classify wallpaper group based on symmetries
        if best_h_score > 0.8 and best_v_score > 0.8:
            return {
                'name': 'p1',  # Simplest wallpaper group
                'notation': 'p1',
                'parameters': {
                    'h_translation': best_h_translation,
                    'v_translation': best_v_translation,
                    'h_score': best_h_score,
                    'v_score': best_v_score
                },
                'confidence': min(best_h_score, best_v_score)
            }
        
        return None
    
    def _calculate_box_counting_dimension(self, image: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        
        # Binarize image
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Box counting
        sizes = [2, 4, 8, 16, 32, 64]
        counts = []
        
        for size in sizes:
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
            
            return max(1.0, min(2.0, fractal_dim))  # Clamp to reasonable range
        
        return 1.0
    
    def _analyze_self_similarity(self, image: np.ndarray) -> Dict:
        """Analyze self-similarity at different scales"""
        
        h, w = image.shape
        similarities = []
        scales = [0.5, 0.25, 0.125]
        
        for scale in scales:
            # Resize image
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(image, (new_w, new_h))
            
            # Find best match in original image
            if scaled.shape[0] < h and scaled.shape[1] < w:
                result = cv2.matchTemplate(image, scaled, cv2.TM_CCOEFF_NORMED)
                max_similarity = np.max(result)
                similarities.append(max_similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        return {
            'score': avg_similarity,
            'scales': scales,
            'similarities': similarities
        }
    
    def _calculate_euler_characteristic(self, image: np.ndarray) -> int:
        """Calculate Euler characteristic (χ = V - E + F)"""
        
        # Simplified calculation using connected components and holes
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Count connected components (faces)
        num_labels, labels = cv2.connectedComponents(binary)
        faces = num_labels - 1  # Subtract background
        
        # Count holes (approximate)
        inverted = 255 - binary
        num_holes, _ = cv2.connectedComponents(inverted)
        holes = num_holes - 1  # Subtract background
        
        # Simplified Euler characteristic
        # For 2D: χ = components - holes
        euler_char = faces - holes
        
        return euler_char
    
    def _analyze_connectivity(self, image: np.ndarray) -> Dict:
        """Analyze topological connectivity"""
        
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Count connected components
        num_components, _ = cv2.connectedComponents(binary)
        components = num_components - 1  # Subtract background
        
        # Count holes using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours and count holes
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        holes = 0
        if hierarchy is not None:
            # Count internal contours (holes)
            for i in range(len(hierarchy[0])):
                if hierarchy[0][i][3] != -1:  # Has parent (is a hole)
                    holes += 1
        
        return {
            'components': components,
            'holes': holes,
            'genus': holes  # For 2D, genus ≈ number of holes
        }
    
    def _find_dominant_frequencies(self, fft_magnitude: np.ndarray) -> List[Tuple[int, int]]:
        """Find dominant frequencies in FFT"""
        
        # Find peaks in frequency domain
        threshold = np.percentile(fft_magnitude, 95)
        peaks = np.where(fft_magnitude > threshold)
        
        # Return frequency coordinates
        return list(zip(peaks[0], peaks[1]))
    
    def _analyze_harmonic_ratios(self, frequencies: List[Tuple[int, int]]) -> Dict:
        """Analyze harmonic relationships between frequencies"""
        
        if len(frequencies) < 2:
            return {'is_harmonic': False, 'confidence': 0.0}
        
        # Calculate frequency magnitudes
        freq_mags = [np.sqrt(f[0]**2 + f[1]**2) for f in frequencies]
        freq_mags.sort()
        
        # Check for harmonic relationships
        ratios = []
        for i in range(1, len(freq_mags)):
            ratio = freq_mags[i] / freq_mags[0]
            ratios.append(ratio)
        
        # Check if ratios are close to integers (harmonic)
        integer_errors = [abs(r - round(r)) for r in ratios]
        avg_error = np.mean(integer_errors)
        
        is_harmonic = avg_error < 0.2
        confidence = 1.0 - avg_error if is_harmonic else 0.0
        
        return {
            'is_harmonic': is_harmonic,
            'ratios': ratios,
            'confidence': confidence,
            'fundamental': freq_mags[0] if freq_mags else 0
        }
    
    def _calculate_radial_spectrum(self, fft_magnitude: np.ndarray) -> np.ndarray:
        """Calculate radial frequency spectrum"""
        
        h, w = fft_magnitude.shape
        center_x, center_y = w // 2, h // 2
        
        # Create radial coordinate system
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Calculate radial average
        max_radius = min(center_x, center_y)
        radial_spectrum = []
        
        for radius in range(1, max_radius):
            mask = (r >= radius - 0.5) & (r < radius + 0.5)
            if np.any(mask):
                radial_spectrum.append(np.mean(fft_magnitude[mask]))
        
        return np.array(radial_spectrum)
    
    def _fit_power_law(self, data: np.ndarray) -> Dict:
        """Fit power law to data"""
        
        if len(data) < 3:
            return {'r_squared': 0.0, 'exponent': 0.0, 'coefficient': 0.0}
        
        # Remove zeros and take log
        data = data[data > 0]
        if len(data) < 3:
            return {'r_squared': 0.0, 'exponent': 0.0, 'coefficient': 0.0}
        
        x = np.arange(1, len(data) + 1)
        log_x = np.log(x)
        log_y = np.log(data)
        
        # Linear regression in log space
        coeffs = np.polyfit(log_x, log_y, 1)
        predicted = np.polyval(coeffs, log_x)
        
        # Calculate R-squared
        ss_res = np.sum((log_y - predicted) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'r_squared': max(0, r_squared),
            'exponent': coeffs[0],
            'coefficient': np.exp(coeffs[1])
        }
    
    def _image_to_graph(self, image: np.ndarray) -> Optional[nx.Graph]:
        """Convert image pattern to graph representation"""
        
        # Detect keypoints
        detector = cv2.ORB_create()
        keypoints = detector.detect(image, None)
        
        if len(keypoints) < 3:
            return None
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i, kp in enumerate(keypoints):
            G.add_node(i, pos=(kp.pt[0], kp.pt[1]))
        
        # Add edges based on proximity
        positions = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        
        # Use Delaunay triangulation for natural connectivity
        from scipy.spatial import Delaunay
        
        try:
            tri = Delaunay(positions)
            
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        G.add_edge(simplex[i], simplex[j])
        
        except Exception:
            # Fallback: connect nearby points
            threshold = min(image.shape) / 10
            
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                                 (positions[i][1] - positions[j][1])**2)
                    if dist < threshold:
                        G.add_edge(i, j)
        
        return G
    
    def _calculate_shannon_entropy(self, image: np.ndarray) -> float:
        """Calculate Shannon entropy of image"""
        
        # Calculate histogram
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        
        # Normalize to probabilities
        hist = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return entropy
    
    def _calculate_lz_complexity(self, image: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity"""
        
        # Convert image to binary string
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        binary_string = ''.join(binary.flatten().astype(str))
        
        # Simple LZ complexity approximation
        # Count unique substrings
        substrings = set()
        
        for i in range(len(binary_string)):
            for j in range(i+1, min(i+10, len(binary_string)+1)):  # Limit substring length
                substrings.add(binary_string[i:j])
        
        # Normalize by theoretical maximum
        max_complexity = len(binary_string) * (len(binary_string) + 1) // 2
        complexity = len(substrings) / max_complexity if max_complexity > 0 else 0
        
        return complexity
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        return rotated
    
    def _find_peaks(self, signal: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Find peaks in 1D signal"""
        
        # Simple peak detection
        peaks = []
        
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(i)
        
        return np.array(peaks), {}

def analyze_kolam_mathematics(image_path: str) -> List[MathematicalPrinciple]:
    """Analyze mathematical principles in a Kolam image"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Initialize analyzer
    analyzer = MathematicalAnalyzer()
    
    # Perform analysis
    principles = analyzer.analyze_mathematical_principles(image)
    
    return principles

if __name__ == "__main__":
    # Example usage
    try:
        principles = analyze_kolam_mathematics("data/Rangoli (1).jpg")
        
        print("Mathematical Principles Found:")
        print("=" * 50)
        
        for principle in principles:
            print(f"\n{principle.name}")
            print(f"Description: {principle.description}")
            print(f"Formula: {principle.formula}")
            print(f"Confidence: {principle.confidence:.3f}")
            print(f"Parameters: {principle.parameters}")
            print("-" * 30)
        
        if not principles:
            print("No significant mathematical principles detected.")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure you have Kolam images in the data directory")