"""
KolamAI - Pattern Generation Engine
Recreates Kolam patterns based on mathematical principles and design rules
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import math
from dataclasses import dataclass
from enum import Enum
import random

class SymmetryType(Enum):
    RADIAL = "radial"
    BILATERAL = "bilateral"
    ROTATIONAL = "rotational"
    TRANSLATIONAL = "translational"

@dataclass
class GenerationParams:
    """Parameters for Kolam generation"""
    canvas_size: Tuple[int, int] = (512, 512)
    grid_size: int = 16
    symmetry_type: SymmetryType = SymmetryType.RADIAL
    complexity_level: float = 0.5  # 0.0 to 1.0
    color_scheme: str = "traditional"  # traditional, modern, monochrome
    line_thickness: int = 3
    dot_radius: int = 4
    cultural_style: str = "tamil"  # tamil, andhra, karnataka, kerala

class KolamGenerator:
    """Advanced Kolam pattern generator based on mathematical principles"""
    
    def __init__(self):
        self.canvas = None
        self.grid_points = None
        self.current_params = None
        
        # Traditional Kolam rules and patterns
        self.traditional_motifs = self._load_traditional_motifs()
        self.cultural_rules = self._load_cultural_rules()
        
    def generate_kolam(self, params: GenerationParams) -> np.ndarray:
        """Generate a complete Kolam pattern"""
        
        self.current_params = params
        
        # Initialize canvas
        self.canvas = np.ones((*params.canvas_size, 3), dtype=np.uint8) * 255
        
        # Generate grid structure
        self._generate_grid_structure()
        
        # Apply symmetry-based generation
        if params.symmetry_type == SymmetryType.RADIAL:
            self._generate_radial_pattern()
        elif params.symmetry_type == SymmetryType.BILATERAL:
            self._generate_bilateral_pattern()
        elif params.symmetry_type == SymmetryType.ROTATIONAL:
            self._generate_rotational_pattern()
        else:
            self._generate_translational_pattern()
        
        # Apply cultural styling
        self._apply_cultural_styling()
        
        # Add decorative elements
        self._add_decorative_elements()
        
        return self.canvas
    
    def _generate_grid_structure(self):
        """Generate the underlying dot grid structure"""
        
        params = self.current_params
        h, w = params.canvas_size
        
        # Calculate grid spacing
        spacing_x = w // (params.grid_size + 1)
        spacing_y = h // (params.grid_size + 1)
        
        # Generate grid points
        self.grid_points = []
        for i in range(1, params.grid_size + 1):
            for j in range(1, params.grid_size + 1):
                x = j * spacing_x
                y = i * spacing_y
                self.grid_points.append((x, y))
        
        # Draw grid dots
        color = self._get_color('dot')
        for point in self.grid_points:
            cv2.circle(self.canvas, point, params.dot_radius, color, -1)
    
    def _generate_radial_pattern(self):
        """Generate radial symmetric pattern"""
        
        params = self.current_params
        center = (params.canvas_size[1] // 2, params.canvas_size[0] // 2)
        
        # Number of radial segments based on complexity
        num_segments = int(4 + params.complexity_level * 8)
        angle_step = 2 * np.pi / num_segments
        
        # Generate one segment and replicate
        segment_pattern = self._generate_radial_segment(center, angle_step)
        
        # Replicate segment around center
        for i in range(num_segments):
            angle = i * angle_step
            rotated_pattern = self._rotate_pattern(segment_pattern, center, angle)
            self._draw_pattern_elements(rotated_pattern)
    
    def _generate_radial_segment(self, center: Tuple[int, int], angle_span: float) -> List[Dict]:
        """Generate pattern elements for one radial segment"""
        
        elements = []
        params = self.current_params
        max_radius = min(params.canvas_size) // 3
        
        # Generate concentric circles
        num_circles = int(2 + params.complexity_level * 4)
        for i in range(1, num_circles + 1):
            radius = (max_radius * i) // num_circles
            
            # Add circular arc
            elements.append({
                'type': 'arc',
                'center': center,
                'radius': radius,
                'start_angle': -angle_span/2,
                'end_angle': angle_span/2,
                'thickness': params.line_thickness
            })
            
            # Add radial lines
            if i % 2 == 0:  # Every other circle
                for j in range(3):
                    angle = (j - 1) * angle_span / 4
                    end_x = center[0] + int(radius * np.cos(angle))
                    end_y = center[1] + int(radius * np.sin(angle))
                    
                    elements.append({
                        'type': 'line',
                        'start': center,
                        'end': (end_x, end_y),
                        'thickness': params.line_thickness
                    })
        
        # Add decorative curves
        if params.complexity_level > 0.5:
            elements.extend(self._generate_decorative_curves(center, max_radius, angle_span))
        
        return elements
    
    def _generate_bilateral_pattern(self):
        """Generate bilaterally symmetric pattern"""
        
        params = self.current_params
        h, w = params.canvas_size
        center_x = w // 2
        
        # Generate left half pattern
        left_pattern = self._generate_half_pattern((center_x // 2, h // 2))
        
        # Draw left half
        self._draw_pattern_elements(left_pattern)
        
        # Mirror to right half
        right_pattern = self._mirror_pattern_horizontal(left_pattern, center_x)
        self._draw_pattern_elements(right_pattern)
        
        # Add central axis elements
        self._add_central_axis_elements()
    
    def _generate_half_pattern(self, center: Tuple[int, int]) -> List[Dict]:
        """Generate pattern elements for half of bilateral design"""
        
        elements = []
        params = self.current_params
        
        # Generate flowing curves
        num_curves = int(3 + params.complexity_level * 5)
        
        for i in range(num_curves):
            # Create curved path using nearby grid points
            curve_points = self._select_curve_points(center, i)
            if len(curve_points) >= 3:
                elements.append({
                    'type': 'curve',
                    'points': curve_points,
                    'thickness': params.line_thickness
                })
        
        # Add geometric shapes
        if params.complexity_level > 0.3:
            elements.extend(self._generate_geometric_shapes(center))
        
        return elements
    
    def _generate_rotational_pattern(self):
        """Generate rotationally symmetric pattern (4-fold, 8-fold, etc.)"""
        
        params = self.current_params
        center = (params.canvas_size[1] // 2, params.canvas_size[0] // 2)
        
        # Determine rotation order based on complexity
        rotation_order = 4 if params.complexity_level < 0.5 else 8
        angle_step = 2 * np.pi / rotation_order
        
        # Generate base pattern (1/nth of full pattern)
        base_pattern = self._generate_rotational_base(center, angle_step)
        
        # Replicate with rotations
        for i in range(rotation_order):
            angle = i * angle_step
            rotated_pattern = self._rotate_pattern(base_pattern, center, angle)
            self._draw_pattern_elements(rotated_pattern)
    
    def _generate_rotational_base(self, center: Tuple[int, int], angle_span: float) -> List[Dict]:
        """Generate base pattern for rotational symmetry"""
        
        elements = []
        params = self.current_params
        
        # Create petal-like shapes
        petal_length = min(params.canvas_size) // 4
        
        # Main petal curve
        petal_points = []
        num_points = 10
        for i in range(num_points):
            t = i / (num_points - 1)
            # Petal curve equation
            r = petal_length * (1 - t) * np.sin(np.pi * t)
            angle = angle_span * (t - 0.5) * 0.8
            
            x = center[0] + int(r * np.cos(angle))
            y = center[1] + int(r * np.sin(angle))
            petal_points.append((x, y))
        
        elements.append({
            'type': 'curve',
            'points': petal_points,
            'thickness': params.line_thickness
        })
        
        # Add inner decorations
        if params.complexity_level > 0.4:
            inner_radius = petal_length // 3
            inner_points = []
            for i in range(5):
                t = i / 4
                angle = angle_span * (t - 0.5) * 0.5
                x = center[0] + int(inner_radius * np.cos(angle))
                y = center[1] + int(inner_radius * np.sin(angle))
                inner_points.append((x, y))
            
            elements.append({
                'type': 'curve',
                'points': inner_points,
                'thickness': params.line_thickness - 1
            })
        
        return elements
    
    def _generate_translational_pattern(self):
        """Generate pattern with translational symmetry"""
        
        params = self.current_params
        h, w = params.canvas_size
        
        # Create repeating unit
        unit_size = min(w, h) // 4
        unit_pattern = self._generate_translation_unit(unit_size)
        
        # Tile across canvas
        for y in range(0, h, unit_size):
            for x in range(0, w, unit_size):
                translated_pattern = self._translate_pattern(unit_pattern, (x, y))
                self._draw_pattern_elements(translated_pattern)
    
    def _generate_translation_unit(self, size: int) -> List[Dict]:
        """Generate a single unit for translational tiling"""
        
        elements = []
        params = self.current_params
        center = (size // 2, size // 2)
        
        # Create diamond shape
        diamond_points = [
            (center[0], center[1] - size//3),
            (center[0] + size//3, center[1]),
            (center[0], center[1] + size//3),
            (center[0] - size//3, center[1])
        ]
        
        elements.append({
            'type': 'polygon',
            'points': diamond_points,
            'thickness': params.line_thickness,
            'filled': False
        })
        
        # Add internal pattern
        if params.complexity_level > 0.3:
            # Internal cross
            elements.append({
                'type': 'line',
                'start': (center[0] - size//4, center[1]),
                'end': (center[0] + size//4, center[1]),
                'thickness': params.line_thickness - 1
            })
            elements.append({
                'type': 'line',
                'start': (center[0], center[1] - size//4),
                'end': (center[0], center[1] + size//4),
                'thickness': params.line_thickness - 1
            })
        
        return elements
    
    def _apply_cultural_styling(self):
        """Apply cultural-specific styling rules"""
        
        params = self.current_params
        center = (params.canvas_size[1] // 2, params.canvas_size[0] // 2)
        
        if params.cultural_style == "tamil":
            self._apply_tamil_styling(center)
        elif params.cultural_style == "andhra":
            self._apply_andhra_styling(center)
        elif params.cultural_style == "karnataka":
            self._apply_karnataka_styling(center)
        elif params.cultural_style == "kerala":
            self._apply_kerala_styling(center)
    
    def _apply_tamil_styling(self, center: Tuple[int, int]):
        """Apply Tamil Kolam specific styling - continuous lines and loops"""
        
        # Tamil Kolams: Continuous lines connecting dots
        self._add_tamil_continuous_lines(center)
        
        # Add traditional Tamil lotus motif
        if self.current_params.complexity_level > 0.5:
            self._add_lotus_motif(center)
    
    def _apply_andhra_styling(self, center: Tuple[int, int]):
        """Apply Andhra Muggu specific styling - geometric patterns"""
        
        # Andhra Muggus: Strong geometric emphasis with borders
        self._add_geometric_border(center)
        
        # Add angular geometric patterns
        self._add_angular_patterns(center)
    
    def _apply_karnataka_styling(self, center: Tuple[int, int]):
        """Apply Karnataka Rangoli specific styling - colorful floral"""
        
        # Karnataka Rangolis: Floral elements and vibrant colors
        self._add_floral_petals(center)
        
        # Add decorative vine patterns
        if self.current_params.complexity_level > 0.4:
            self._add_vine_patterns(center)
    
    def _apply_kerala_styling(self, center: Tuple[int, int]):
        """Apply Kerala specific styling - organic flowing lines"""
        
        # Kerala patterns: Organic curves and nature motifs
        self._add_organic_curves(center)
        
        # Add traditional Kerala lamp motif
        if self.current_params.complexity_level > 0.6:
            self._add_lamp_motif(center)
    
    def _add_decorative_elements(self):
        """Add final decorative touches"""
        
        params = self.current_params
        
        if params.complexity_level > 0.7:
            # Add small decorative dots
            self._add_decorative_dots()
            
            # Add fine line details
            self._add_fine_details()
        
        # Add border if needed
        if params.complexity_level > 0.5:
            self._add_outer_border()
    
    def _draw_pattern_elements(self, elements: List[Dict]):
        """Draw pattern elements on canvas"""
        
        for element in elements:
            if element['type'] == 'line':
                self._draw_line(element)
            elif element['type'] == 'curve':
                self._draw_curve(element)
            elif element['type'] == 'arc':
                self._draw_arc(element)
            elif element['type'] == 'polygon':
                self._draw_polygon(element)
            elif element['type'] == 'circle':
                self._draw_circle(element)
    
    def _draw_line(self, element: Dict):
        """Draw a line element"""
        color = self._get_color('line')
        cv2.line(self.canvas, element['start'], element['end'], 
                color, element['thickness'])
    
    def _draw_curve(self, element: Dict):
        """Draw a smooth curve through points"""
        if len(element['points']) < 2:
            return
            
        color = self._get_color('curve')
        points = np.array(element['points'], dtype=np.int32)
        
        # Draw smooth curve using polylines
        for i in range(len(points) - 1):
            cv2.line(self.canvas, tuple(points[i]), tuple(points[i+1]), 
                    color, element['thickness'])
    
    def _draw_arc(self, element: Dict):
        """Draw an arc element"""
        color = self._get_color('arc')
        center = element['center']
        radius = element['radius']
        start_angle = int(element['start_angle'] * 180 / np.pi)
        end_angle = int(element['end_angle'] * 180 / np.pi)
        
        cv2.ellipse(self.canvas, center, (radius, radius), 0, 
                   start_angle, end_angle, color, element['thickness'])
    
    def _draw_polygon(self, element: Dict):
        """Draw a polygon element"""
        color = self._get_color('polygon')
        points = np.array(element['points'], dtype=np.int32)
        
        if element.get('filled', False):
            cv2.fillPoly(self.canvas, [points], color)
        else:
            cv2.polylines(self.canvas, [points], True, color, element['thickness'])
    
    def _draw_circle(self, element: Dict):
        """Draw a circle element"""
        color = self._get_color('circle')
        cv2.circle(self.canvas, element['center'], element['radius'], 
                  color, element.get('thickness', -1))
    
    def _get_color(self, element_type: str) -> Tuple[int, int, int]:
        """Get color for element based on color scheme"""
        
        scheme = self.current_params.color_scheme
        
        if scheme == "monochrome":
            return (0, 0, 0)  # Black
        elif scheme == "traditional":
            colors = {
                'line': (139, 69, 19),    # Brown
                'curve': (255, 0, 0),     # Red
                'arc': (255, 165, 0),     # Orange
                'polygon': (0, 128, 0),   # Green
                'circle': (128, 0, 128),  # Purple
                'dot': (0, 0, 0)          # Black
            }
            return colors.get(element_type, (0, 0, 0))
        else:  # modern
            colors = {
                'line': (70, 130, 180),   # Steel Blue
                'curve': (220, 20, 60),   # Crimson
                'arc': (255, 140, 0),     # Dark Orange
                'polygon': (34, 139, 34), # Forest Green
                'circle': (138, 43, 226), # Blue Violet
                'dot': (25, 25, 112)      # Midnight Blue
            }
            return colors.get(element_type, (0, 0, 0))
    
    # Helper methods for pattern transformations
    def _rotate_pattern(self, pattern: List[Dict], center: Tuple[int, int], angle: float) -> List[Dict]:
        """Rotate pattern elements around center"""
        rotated = []
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        for element in pattern:
            new_element = element.copy()
            
            if 'start' in element and 'end' in element:
                # Rotate line endpoints
                new_element['start'] = self._rotate_point(element['start'], center, cos_a, sin_a)
                new_element['end'] = self._rotate_point(element['end'], center, cos_a, sin_a)
            elif 'points' in element:
                # Rotate all points
                new_element['points'] = [
                    self._rotate_point(p, center, cos_a, sin_a) for p in element['points']
                ]
            elif 'center' in element:
                # Rotate center point
                new_element['center'] = self._rotate_point(element['center'], center, cos_a, sin_a)
                if 'start_angle' in element:
                    new_element['start_angle'] = element['start_angle'] + angle
                    new_element['end_angle'] = element['end_angle'] + angle
            
            rotated.append(new_element)
        
        return rotated
    
    def _rotate_point(self, point: Tuple[int, int], center: Tuple[int, int], 
                     cos_a: float, sin_a: float) -> Tuple[int, int]:
        """Rotate a single point around center"""
        x, y = point[0] - center[0], point[1] - center[1]
        new_x = x * cos_a - y * sin_a + center[0]
        new_y = x * sin_a + y * cos_a + center[1]
        return (int(new_x), int(new_y))
    
    def _mirror_pattern_horizontal(self, pattern: List[Dict], axis_x: int) -> List[Dict]:
        """Mirror pattern horizontally across vertical axis"""
        mirrored = []
        
        for element in pattern:
            new_element = element.copy()
            
            if 'start' in element and 'end' in element:
                new_element['start'] = (2 * axis_x - element['start'][0], element['start'][1])
                new_element['end'] = (2 * axis_x - element['end'][0], element['end'][1])
            elif 'points' in element:
                new_element['points'] = [
                    (2 * axis_x - p[0], p[1]) for p in element['points']
                ]
            elif 'center' in element:
                new_element['center'] = (2 * axis_x - element['center'][0], element['center'][1])
            
            mirrored.append(new_element)
        
        return mirrored
    
    def _translate_pattern(self, pattern: List[Dict], offset: Tuple[int, int]) -> List[Dict]:
        """Translate pattern by offset"""
        translated = []
        dx, dy = offset
        
        for element in pattern:
            new_element = element.copy()
            
            if 'start' in element and 'end' in element:
                new_element['start'] = (element['start'][0] + dx, element['start'][1] + dy)
                new_element['end'] = (element['end'][0] + dx, element['end'][1] + dy)
            elif 'points' in element:
                new_element['points'] = [
                    (p[0] + dx, p[1] + dy) for p in element['points']
                ]
            elif 'center' in element:
                new_element['center'] = (element['center'][0] + dx, element['center'][1] + dy)
            
            translated.append(new_element)
        
        return translated
    
    # Placeholder methods for advanced features
    def _load_traditional_motifs(self) -> Dict:
        """Load traditional motif patterns"""
        return {
            'tamil': ['lotus', 'peacock', 'lamp'],
            'andhra': ['flower', 'geometric', 'bird'],
            'karnataka': ['creeper', 'mandala', 'star'],
            'kerala': ['elephant', 'boat', 'coconut']
        }
    
    def _load_cultural_rules(self) -> Dict:
        """Load cultural-specific rules"""
        return {
            'tamil': {'continuous_lines': True, 'closed_loops': True},
            'andhra': {'geometric_emphasis': True, 'border_required': True},
            'karnataka': {'color_variety': True, 'floral_elements': True},
            'kerala': {'organic_curves': True, 'nature_motifs': True}
        }
    
    def _select_curve_points(self, center: Tuple[int, int], curve_index: int) -> List[Tuple[int, int]]:
        """Select grid points for creating curves"""
        # Simple implementation - select nearby grid points
        nearby_points = []
        for point in self.grid_points:
            distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
            if 50 < distance < 150:  # Within reasonable range
                nearby_points.append(point)
        
        # Return subset based on curve index
        if len(nearby_points) > 3:
            start_idx = (curve_index * 2) % len(nearby_points)
            return nearby_points[start_idx:start_idx+4]
        return nearby_points
    
    def _generate_decorative_curves(self, center: Tuple[int, int], 
                                  max_radius: int, angle_span: float) -> List[Dict]:
        """Generate decorative curve elements"""
        curves = []
        # Simple spiral curve
        points = []
        for i in range(20):
            t = i / 19
            r = max_radius * 0.3 * (1 - t)
            angle = angle_span * t * 2
            x = center[0] + int(r * np.cos(angle))
            y = center[1] + int(r * np.sin(angle))
            points.append((x, y))
        
        curves.append({
            'type': 'curve',
            'points': points,
            'thickness': self.current_params.line_thickness - 1
        })
        
        return curves
    
    def _generate_geometric_shapes(self, center: Tuple[int, int]) -> List[Dict]:
        """Generate geometric shape elements"""
        shapes = []
        size = 30
        
        # Triangle
        triangle_points = [
            (center[0], center[1] - size),
            (center[0] - size, center[1] + size//2),
            (center[0] + size, center[1] + size//2)
        ]
        
        shapes.append({
            'type': 'polygon',
            'points': triangle_points,
            'thickness': self.current_params.line_thickness,
            'filled': False
        })
        
        return shapes
    
    # Cultural styling implementations
    def _add_tamil_continuous_lines(self, center: Tuple[int, int]):
        """Add Tamil-style continuous connecting lines"""
        color = self._get_color('curve')
        
        # Connect nearby grid points with flowing curves
        for i in range(0, len(self.grid_points) - 1, 2):
            if i + 1 < len(self.grid_points):
                p1 = self.grid_points[i]
                p2 = self.grid_points[i + 1]
                
                # Create curved connection
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2 - 20
                
                cv2.line(self.canvas, p1, (mid_x, mid_y), color, self.current_params.line_thickness)
                cv2.line(self.canvas, (mid_x, mid_y), p2, color, self.current_params.line_thickness)
    
    def _add_lotus_motif(self, center: Tuple[int, int]):
        """Add Tamil lotus motif"""
        color = self._get_color('curve')
        radius = 40
        
        # Draw lotus petals
        for i in range(8):
            angle = i * np.pi / 4
            petal_end = (
                center[0] + int(radius * np.cos(angle)),
                center[1] + int(radius * np.sin(angle))
            )
            
            # Draw petal as ellipse
            cv2.ellipse(self.canvas, center, (radius//2, radius//3), 
                       int(angle * 180 / np.pi), 0, 180, color, 2)
    
    def _add_geometric_border(self, center: Tuple[int, int]):
        """Add Andhra-style geometric border"""
        color = self._get_color('polygon')
        h, w = self.current_params.canvas_size
        
        # Draw rectangular border with geometric patterns
        border_thickness = 20
        
        # Outer rectangle
        cv2.rectangle(self.canvas, (border_thickness, border_thickness), 
                     (w - border_thickness, h - border_thickness), color, 3)
        
        # Add geometric triangular patterns along border
        for i in range(border_thickness, w - border_thickness, 40):
            triangle_points = np.array([
                [i, border_thickness],
                [i + 15, border_thickness - 10],
                [i + 30, border_thickness]
            ], np.int32)
            cv2.polylines(self.canvas, [triangle_points], True, color, 2)
    
    def _add_angular_patterns(self, center: Tuple[int, int]):
        """Add Andhra-style angular geometric patterns"""
        color = self._get_color('line')
        
        # Create diamond patterns around center
        for radius in [60, 90, 120]:
            diamond_points = [
                (center[0], center[1] - radius),
                (center[0] + radius, center[1]),
                (center[0], center[1] + radius),
                (center[0] - radius, center[1])
            ]
            
            points = np.array(diamond_points, np.int32)
            cv2.polylines(self.canvas, [points], True, color, 2)
    
    def _add_floral_petals(self, center: Tuple[int, int]):
        """Add Karnataka-style floral petals"""
        colors = [(0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]  # Bright colors
        
        # Draw colorful flower petals
        for i in range(6):
            angle = i * np.pi / 3
            petal_center = (
                center[0] + int(50 * np.cos(angle)),
                center[1] + int(50 * np.sin(angle))
            )
            
            color = colors[i % len(colors)]
            cv2.circle(self.canvas, petal_center, 25, color, -1)
            cv2.circle(self.canvas, petal_center, 25, (0, 0, 0), 2)
    
    def _add_vine_patterns(self, center: Tuple[int, int]):
        """Add Karnataka-style decorative vine patterns"""
        color = (0, 128, 0)  # Green for vines
        
        # Create flowing vine-like curves
        for angle_offset in [0, np.pi/2, np.pi, 3*np.pi/2]:
            points = []
            for i in range(20):
                t = i / 19
                radius = 80 + 30 * np.sin(t * 4 * np.pi)
                angle = angle_offset + t * np.pi
                
                x = center[0] + int(radius * np.cos(angle))
                y = center[1] + int(radius * np.sin(angle))
                points.append((x, y))
            
            # Draw vine
            for i in range(len(points) - 1):
                cv2.line(self.canvas, points[i], points[i+1], color, 3)
    
    def _add_organic_curves(self, center: Tuple[int, int]):
        """Add Kerala-style organic flowing curves"""
        color = self._get_color('curve')
        
        # Create organic, wave-like patterns
        for radius_base in [70, 100, 130]:
            points = []
            for i in range(50):
                t = i / 49
                angle = t * 2 * np.pi
                
                # Add organic variation to radius
                radius = radius_base + 20 * np.sin(t * 6 * np.pi) * np.cos(t * 4 * np.pi)
                
                x = center[0] + int(radius * np.cos(angle))
                y = center[1] + int(radius * np.sin(angle))
                points.append((x, y))
            
            # Draw organic curve
            points_array = np.array(points, np.int32)
            cv2.polylines(self.canvas, [points_array], True, color, 2)
    
    def _add_lamp_motif(self, center: Tuple[int, int]):
        """Add Kerala traditional lamp motif"""
        color = (0, 165, 255)  # Orange for lamp
        
        # Draw lamp base
        base_points = np.array([
            [center[0] - 30, center[1] + 40],
            [center[0] + 30, center[1] + 40],
            [center[0] + 20, center[1] + 10],
            [center[0] - 20, center[1] + 10]
        ], np.int32)
        
        cv2.fillPoly(self.canvas, [base_points], color)
        
        # Draw flame
        flame_points = np.array([
            [center[0], center[1] - 20],
            [center[0] - 10, center[1]],
            [center[0] + 10, center[1]]
        ], np.int32)
        
        cv2.fillPoly(self.canvas, [flame_points], (0, 0, 255))  # Red flame
    
    # Placeholder implementations for remaining methods
    def _add_decorative_dots(self): 
        """Add small decorative dots around the pattern"""
        color = self._get_color('dot')
        h, w = self.current_params.canvas_size
        
        # Add random decorative dots
        for _ in range(int(self.current_params.complexity_level * 20)):
            x = random.randint(20, w - 20)
            y = random.randint(20, h - 20)
            cv2.circle(self.canvas, (x, y), 2, color, -1)
    
    def _add_fine_details(self): 
        """Add fine line details"""
        color = self._get_color('line')
        center = (self.current_params.canvas_size[1] // 2, self.current_params.canvas_size[0] // 2)
        
        # Add fine radial lines
        for i in range(16):
            angle = i * np.pi / 8
            start_radius = 30
            end_radius = 60
            
            start_point = (
                center[0] + int(start_radius * np.cos(angle)),
                center[1] + int(start_radius * np.sin(angle))
            )
            end_point = (
                center[0] + int(end_radius * np.cos(angle)),
                center[1] + int(end_radius * np.sin(angle))
            )
            
            cv2.line(self.canvas, start_point, end_point, color, 1)
    
    def _add_outer_border(self): 
        """Add outer decorative border"""
        color = self._get_color('polygon')
        h, w = self.current_params.canvas_size
        
        # Simple decorative border
        cv2.rectangle(self.canvas, (10, 10), (w - 10, h - 10), color, 2)
    
    def _add_central_axis_elements(self): 
        """Add elements along central axis for bilateral patterns"""
        color = self._get_color('line')
        h, w = self.current_params.canvas_size
        center_x = w // 2
        
        # Draw central vertical line with decorations
        cv2.line(self.canvas, (center_x, 50), (center_x, h - 50), color, 2)
        
        # Add decorative elements along the axis
        for y in range(100, h - 100, 60):
            cv2.circle(self.canvas, (center_x, y), 8, color, 2)

def generate_kolam_variations(base_params: GenerationParams, num_variations: int = 5) -> List[np.ndarray]:
    """Generate multiple variations of a Kolam pattern"""
    
    generator = KolamGenerator()
    variations = []
    
    for i in range(num_variations):
        # Create variation by modifying parameters
        params = GenerationParams(
            canvas_size=base_params.canvas_size,
            grid_size=base_params.grid_size + random.randint(-2, 2),
            symmetry_type=base_params.symmetry_type,
            complexity_level=max(0.1, min(1.0, base_params.complexity_level + random.uniform(-0.2, 0.2))),
            color_scheme=base_params.color_scheme,
            line_thickness=max(1, base_params.line_thickness + random.randint(-1, 1)),
            dot_radius=max(2, base_params.dot_radius + random.randint(-1, 1)),
            cultural_style=base_params.cultural_style
        )
        
        kolam = generator.generate_kolam(params)
        variations.append(kolam)
    
    return variations

if __name__ == "__main__":
    # Example usage
    params = GenerationParams(
        canvas_size=(512, 512),
        grid_size=12,
        symmetry_type=SymmetryType.RADIAL,
        complexity_level=0.7,
        color_scheme="traditional",
        cultural_style="tamil"
    )
    
    generator = KolamGenerator()
    kolam = generator.generate_kolam(params)
    
    # Save generated Kolam
    cv2.imwrite("generated_kolam.png", kolam)
    
    # Display
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(kolam, cv2.COLOR_BGR2RGB))
    plt.title("Generated Kolam Pattern")
    plt.axis('off')
    plt.show()
    
    print("Kolam generated successfully!")