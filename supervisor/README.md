# Computer Vision Supervisor Package

This package provides computer vision capabilities for Atari game object detection.

## Package Structure

- `detector.py`: Base class for computer vision operations
- `object_detector.py`: Game object detection functionality
- `pill_detector.py`: Specialized detector for pills in Pacman-like games
- `__init__.py`: Package initialization and public interface

## Classes

### CVSupervisor (Base Class)
Base class providing common functionality for computer vision operations.

### ObjectDetector
Detects and classifies game objects based on color clustering.

Key methods:
- `extract_multiple_colors_clusters()`: Extract clusters of specified colors
- `classify_game_objects()`: Classify objects by size (ghost, pacman, etc.)

### PillDetector
Specialized detector for finding numerous small clusters representing pills.

Key methods:
- `detect_pills()`: Detect numerous small clusters (pills) in the image

## Usage Example

```python
from supervisor import ObjectDetector, PillDetector

# Create detectors
object_detector = ObjectDetector(image, args, iteration, epoch)
pill_detector = PillDetector(image, args, iteration, epoch)

# Analyze image colors
result = object_detector.analyze_env_img()

# Extract object clusters
clusters = object_detector.extract_multiple_colors_clusters(colors, classify_objects=True)

# Detect pills
pill_positions, count = pill_detector.detect_pills(pill_color)
```