"""
Pill Detector for detecting pills in Pacman-like games.

This module provides functionality to detect numerous small clusters in images,
which is particularly useful for finding pills in Pacman-like games.
"""
import cv2
import numpy as np
from .detector import CVSupervisor


class PillDetector(CVSupervisor):
    """
    Pill Detector for Pacman-like Games
    
    This class specializes in detecting numerous small clusters that represent
    pills or dots in games like Pacman. It works by identifying color clusters
    that meet specific size and quantity criteria.
    """

    def detect_pills(self, target_colors, min_area=1, max_area=20, min_count=5, mask=False):
        """
        Detect numerous small clusters in the image (used to find pills in Pacman game)
        
        Args:
            target_colors: Target color list, each color is (R, G, B) or grayscale value.
                          If False, detect pills purely based on area logic without color filtering.
            min_area: Minimum cluster area
            max_area: Maximum cluster area
            min_count: Minimum cluster count threshold
            mask: If True, also return mask of all detected pills
            
        Returns:
            pill_positions: Pill center positions list [(x, y), ...]
            clusters_count: Total number of detected clusters
            pill_masks: Mask of all detected pills (returned only if mask=True)
        """
        # Preprocess image
        processed_env_img = self.preprocess_env_img()
        
        # Use original BGR image for processing
        img_bgr = processed_env_img
        
        # Collect all contours
        all_filtered_contours = []
        
        if target_colors is False:
            # Detect pills purely based on area logic without color filtering
            # Convert image to grayscale
            if len(img_bgr.shape) == 3:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_bgr
                
            # Create binary mask using Otsu's threshold
            _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours (i.e. clusters) with full hierarchy
            contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # For very small contours, also consider their bounding rectangle dimensions
                if area == 0:
                    # If area is zero, check bounding rectangle dimensions
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Consider as valid if either dimension is within range
                    if min_area <= max(w, h) <= max_area:
                        all_filtered_contours.append(cnt)
                elif min_area <= area <= max_area:
                    all_filtered_contours.append(cnt)
        else:
            # Handle single color as a list
            if not isinstance(target_colors, (list, tuple)) or (
                    len(target_colors) > 0 and not hasattr(target_colors[0], '__len__')):
                target_colors = [target_colors]
            
            # Create masks and find clusters for each color
            for target_color in target_colors:
                # Create mask for specified color (note OpenCV uses BGR format)
                try:
                    if len(img_bgr.shape) == 3 and hasattr(target_color, '__len__') and len(target_color) == 3:
                        # Convert RGB color to BGR
                        bgr_color = np.array([target_color[2], target_color[1], target_color[0]], dtype=np.uint8)
                        mask_color = cv2.inRange(img_bgr, bgr_color, bgr_color)
                    else:
                        mask_color = cv2.inRange(img_bgr, np.array(target_color, dtype=np.uint8), 
                                          np.array(target_color, dtype=np.uint8))
                    
                    # Find contours (i.e. clusters) with full hierarchy
                    contours, hierarchy = cv2.findContours(mask_color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Filter contours by area and add to the collective list
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        # For very small contours, also consider their bounding rectangle dimensions
                        if area == 0:
                            # If area is zero, check bounding rectangle dimensions
                            x, y, w, h = cv2.boundingRect(cnt)
                            # Consider as valid if either dimension is within range
                            if min_area <= max(w, h) <= max_area:
                                all_filtered_contours.append(cnt)
                        elif min_area <= area <= max_area:
                            all_filtered_contours.append(cnt)
                            
                except Exception as e:
                    print(f"Warning: Error processing color {target_color}: {e}")
                    continue
        
        # If cluster count is not enough, consider it not pills
        if len(all_filtered_contours) < min_count:
            if mask:
                # Return empty mask with same shape as input image
                empty_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
                return [], 0, empty_mask
            else:
                return [], 0
        
        # Calculate center position for each cluster
        pill_positions = []
        for contour in all_filtered_contours:
            # Calculate moments of contour
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                # Calculate center point coordinates
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                pill_positions.append((cx, cy))
            else:
                # For contours with zero moments, use bounding rectangle center
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2
                pill_positions.append((cx, cy))
        
        # Create mask of all detected pills if requested
        if mask:
            pill_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
            cv2.drawContours(pill_mask, all_filtered_contours, -1, 255, -1)
        
        # Save result image if capture is enabled
        if hasattr(self.args, 'capture') and self.args.capture:
            # Draw pill positions on image
            annotated_image = self._draw_pill_positions(processed_env_img, pill_positions)
            cv2.imwrite(f"pill_detection_result_{self.epoch}_iter_{self.iter}.png", annotated_image)
        
        if mask:
            return pill_positions, len(all_filtered_contours), pill_mask
        else:
            return pill_positions, len(all_filtered_contours)

    def _draw_pill_positions(self, image, pill_positions):
        """
        Draw pill positions on image
        
        Args:
            image: Input image
            pill_positions: Pill positions list [(x, y), ...]
            
        Returns:
            annotated_image: Image with pill positions marked
        """
        # Copy image to avoid modifying original
        annotated_image = image.copy()
        
        # Mark each pill position with a green dot
        for (x, y) in pill_positions:
            cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)
        
        return annotated_image