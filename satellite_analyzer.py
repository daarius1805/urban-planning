"""
Satellite Image Analyzer for UrbanPlannerAI
Analyzes satellite images to detect greenery using ML
"""

import numpy as np
from PIL import Image
import requests
from io import BytesIO

class SatelliteAnalyzer:
    """Analyze satellite imagery for urban greenery detection"""
    
    def __init__(self):
        self.ndvi_threshold = 0.3  # Threshold for vegetation
    
    def calculate_ndvi_from_rgb(self, image):
        """
        Calculate NDVI-like metric from RGB satellite image
        
        NDVI typically uses Near-Infrared (NIR) and Red bands:
        NDVI = (NIR - Red) / (NIR + Red)
        
        For RGB images, we approximate using Green and Red:
        Pseudo-NDVI = (Green - Red) / (Green + Red)
        
        Args:
            image: PIL Image or numpy array (H, W, 3)
        
        Returns:
            ndvi_score: Greenery percentage (0-100)
            ndvi_map: 2D array of NDVI values
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Normalize to 0-1
        img_array = img_array.astype(float) / 255.0
        
        # Extract RGB channels
        red = img_array[:, :, 0]
        green = img_array[:, :, 1]
        blue = img_array[:, :, 2]
        
        # Calculate pseudo-NDVI
        # Avoid division by zero
        denominator = green + red + 1e-8
        ndvi_map = (green - red) / denominator
        
        # Alternative: Use green intensity as proxy
        # High green = likely vegetation
        green_intensity = green - 0.5 * (red + blue)
        
        # Threshold for vegetation detection
        vegetation_mask = (ndvi_map > self.ndvi_threshold) | (green_intensity > 0.2)
        
        # Calculate percentage
        greenery_percent = (vegetation_mask.sum() / vegetation_mask.size) * 100
        
        return greenery_percent, ndvi_map, vegetation_mask
    
    def analyze_from_url(self, url):
        """
        Download and analyze satellite image from URL
        
        Args:
            url: Direct URL to satellite image (JPEG/PNG)
        
        Returns:
            dict with analysis results
        """
        try:
            response = requests.get(url, timeout=30)
            image = Image.open(BytesIO(response.content))
            
            greenery_percent, ndvi_map, mask = self.calculate_ndvi_from_rgb(image)
            
            return {
                'greenery_percent': greenery_percent,
                'ndvi_map': ndvi_map,
                'vegetation_mask': mask,
                'image': image,
                'success': True
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_from_file(self, filepath):
        """
        Analyze satellite image from local file
        
        Args:
            filepath: Path to image file
        
        Returns:
            dict with analysis results
        """
        try:
            image = Image.open(filepath)
            greenery_percent, ndvi_map, mask = self.calculate_ndvi_from_rgb(image)
            
            return {
                'greenery_percent': greenery_percent,
                'ndvi_map': ndvi_map,
                'vegetation_mask': mask,
                'image': image,
                'success': True
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_google_static_map(self, lat, lon, zoom=16, size='640x640'):
        """
        Get satellite view from Google Static Maps API
        
        Note: Requires Google Cloud API key (free tier available)
        Sign up: https://console.cloud.google.com/
        Enable: Maps Static API
        
        Args:
            lat, lon: Coordinates
            zoom: Zoom level (1-20, higher = more detail)
            size: Image size (max 640x640 for free tier)
        
        Returns:
            PIL Image
        """
        # Placeholder - add your API key
        api_key = "YOUR_GOOGLE_MAPS_API_KEY"
        
        url = (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={lat},{lon}&zoom={zoom}&size={size}"
            f"&maptype=satellite&key={api_key}"
        )
        
        try:
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            print(f"Error fetching map: {e}")
            return None
    
    def batch_analyze_coordinates(self, coordinates_df):
        """
        Analyze multiple locations from DataFrame
        
        Args:
            coordinates_df: DataFrame with 'latitude' and 'longitude' columns
        
        Returns:
            DataFrame with added 'greenery_percent' column
        """
        results = []
        
        for idx, row in coordinates_df.iterrows():
            # Get satellite image
            image = self.get_google_static_map(row['latitude'], row['longitude'])
            
            if image:
                # Analyze greenery
                greenery, _, _ = self.calculate_ndvi_from_rgb(image)
                results.append(greenery)
            else:
                results.append(np.nan)
        
        coordinates_df['ml_greenery_percent'] = results
        return coordinates_df
    
    def visualize_analysis(self, image, ndvi_map, mask):
        """
        Create visualization of analysis results
        
        Returns:
            Combined visualization image
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Satellite Image')
        axes[0].axis('off')
        
        # NDVI h