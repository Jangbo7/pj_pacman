"""
Base class for computer vision based Atari game object detection.
"""
import cv2
import numpy as np


class CVSupervisor:
    """
    Computer Vision Supervisor for Atari Game Object Detection
    """

    def __init__(self, env_img, args, iter_num, epoch):
        """
        Initialize the CV Supervisor
        
        Args:
            env_img: Input game environment image
            args: Command line arguments
            iter_num: Iteration number
            epoch: Epoch number
        """
        self.env_img = env_img
        self.args = args
        self.iter = iter_num
        self.epoch = epoch

    def preprocess_env_img(self):
        """
        Preprocess the environment image
        
        Returns:
            processed_env_img: Processed environment image
        """
        # Resize the image to the specified size
        processed_env_img = cv2.resize(self.env_img, (self.args.size, self.args.size))
        
        # Save the processed image if capture is enabled
        if hasattr(self.args, 'capture') and self.args.capture:
            cv2.imwrite(f"env_img_epoch_{self.epoch}_iter_{self.iter}.png", processed_env_img)
        
        return processed_env_img

    def analyze_env_img(self):
        """
        Analyze the image to find the top 10 main colors and generate a color chart
        
        Returns:
            analysis_result: Color analysis result including main color RGB values and color chart image
        """
        # Preprocess the image if not already done
        processed_env_img = self.preprocess_env_img()
        
        # Convert image to RGB format (if currently in BGR format)
        if len(processed_env_img.shape) == 3 and processed_env_img.shape[2] == 3:
            img_rgb = cv2.cvtColor(processed_env_img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = processed_env_img
        
        # Get all unique colors in the image
        if len(img_rgb.shape) == 3:
            # Color image
            pixels = img_rgb.reshape(-1, 3)
        else:
            # Grayscale image
            pixels = img_rgb.reshape(-1, 1)
        
        # Find unique colors and their counts
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Sort by count in descending order
        sorted_indices = np.argsort(counts)[::-1]
        unique_colors = unique_colors[sorted_indices]
        counts = counts[sorted_indices]
        
        # Get top 10 main colors (or all colors if fewer than 10)
        num_main_colors = min(28, len(unique_colors))
        main_colors = unique_colors[:num_main_colors]
        main_counts = counts[:num_main_colors]
        
        # Build analysis result
        analysis_result = {
            'unique_colors': unique_colors,
            'color_counts': counts,
            'main_colors': main_colors,
            'main_counts': main_counts,
            'processed_image': processed_env_img
        }
        
        return analysis_result