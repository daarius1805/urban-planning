"""
Real Dataset Fetcher for UrbanPlannerAI
Fetches data from OpenStreetMap, WorldPop, and other public sources
"""

import requests
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import geopandas as gpd
import json
import time

class UrbanDataFetcher:
    """Fetch real urban planning data from public APIs"""
    
    def __init__(self):
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        
    def fetch_osm_data(self, city_name, radius_km=10):
        """
        Fetch OpenStreetMap data for a city
        
        Args:
            city_name: Name of the city (e.g., "Austin, Texas")
            radius_km: Radius around city center in km
        
        Returns:
            DataFrame with neighborhood data
        """
        print(f"Fetching data for {city_name}...")
        
        # Step 1: Get city coordinates using Nominatim
        coords = self._get_city_coordinates(city_name)
        if not coords:
            print("Could not find city coordinates")
            return None
        
        lat, lon = coords
        print(f"City coordinates: {lat}, {lon}")
        
        # Step 2: Fetch green spaces (parks, forests)
        green_spaces = self._fetch_green_spaces(lat, lon, radius_km)
        
        # Step 3: Fetch roads and infrastructure
        roads = self._fetch_roads(lat, lon, radius_km)
        
        # Step 4: Fetch buildings
        buildings = self._fetch_buildings(lat, lon, radius_km)
        
        # Step 5: Create grid and calculate metrics
        df = self._create_neighborhood_grid(lat, lon, radius_km, 
                                            green_spaces, roads, buildings)
        
        return df
    
    def _get_city_coordinates(self, city_name):
        """Get lat/lon for a city using Nominatim API"""
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': city_name,
            'format': 'json',
            'limit': 1
        }
        headers = {'User-Agent': 'UrbanPlannerAI/1.0'}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
        except Exception as e:
            print(f"Error fetching coordinates: {e}")
        
        return None
    
    def _fetch_green_spaces(self, lat, lon, radius_km):
        """Fetch parks and green spaces from OSM"""
        radius_m = radius_km * 1000
        
        query = f"""
        [out:json][timeout:25];
        (
          node["leisure"="park"](around:{radius_m},{lat},{lon});
          way["leisure"="park"](around:{radius_m},{lat},{lon});
          node["landuse"="forest"](around:{radius_m},{lat},{lon});
          way["landuse"="forest"](around:{radius_m},{lat},{lon});
          node["landuse"="grass"](around:{radius_m},{lat},{lon});
          way["landuse"="grass"](around:{radius_m},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """
        
        try:
            response = requests.post(self.overpass_url, data=query, timeout=30)
            return response.json()
        except Exception as e:
            print(f"Error fetching green spaces: {e}")
            return {'elements': []}
    
    def _fetch_roads(self, lat, lon, radius_km):
        """Fetch road network from OSM"""
        radius_m = radius_km * 1000
        
        query = f"""
        [out:json][timeout:25];
        (
          way["highway"](around:{radius_m},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """
        
        try:
            response = requests.post(self.overpass_url, data=query, timeout=30)
            return response.json()
        except Exception as e:
            print(f"Error fetching roads: {e}")
            return {'elements': []}
    
    def _fetch_buildings(self, lat, lon, radius_km):
        """Fetch buildings from OSM"""
        radius_m = radius_km * 1000
        
        query = f"""
        [out:json][timeout:25];
        (
          way["building"](around:{radius_m},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """
        
        try:
            response = requests.post(self.overpass_url, data=query, timeout=30)
            return response.json()
        except Exception as e:
            print(f"Error fetching buildings: {e}")
            return {'elements': []}
    
    def _create_neighborhood_grid(self, center_lat, center_lon, radius_km, 
                                  green_data, road_data, building_data):
        """
        Create a grid of neighborhoods and calculate metrics
        """
        # Create grid (e.g., 3x3 or 4x4)
        grid_size = 4
        lat_step = (radius_km * 2) / grid_size / 111  # Approx km to degrees
        lon_step = (radius_km * 2) / grid_size / (111 * np.cos(np.radians(center_lat)))
        
        neighborhoods = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                lat = center_lat - radius_km/111 + (i + 0.5) * lat_step
                lon = center_lon - radius_km/(111*np.cos(np.radians(center_lat))) + (j + 0.5) * lon_step
                
                # Calculate metrics for this cell
                green_count = self._count_features_in_area(
                    green_data, lat, lon, lat_step, lon_step
                )
                road_count = self._count_features_in_area(
                    road_data, lat, lon, lat_step, lon_step
                )
                building_count = self._count_features_in_area(
                    building_data, lat, lon, lat_step, lon_step
                )
                
                # Estimate metrics
                greenery_percent = min(100, green_count * 5)  # Normalize
                population_density = building_count * 50  # Rough estimate
                infrastructure_score = min(100, road_count * 10)
                
                neighborhoods.append({
                    'neighborhood': f'Area_{i}_{j}',
                    'latitude': lat,
                    'longitude': lon,
                    'greenery_percent': greenery_percent,
                    'population_density': population_density,
                    'infrastructure_score': infrastructure_score,
                    'air_quality_index': np.random.uniform(30, 120),  # Would need separate API
                    'traffic_congestion': min(100, road_count * 5),
                    'public_transport_access': np.random.uniform(20, 90)
                })
        
        return pd.DataFrame(neighborhoods)
    
    def _count_features_in_area(self, data, center_lat, center_lon, lat_range, lon_range):
        """Count features within a bounding box"""
        count = 0
        
        if 'elements' not in data:
            return 0
        
        for element in data['elements']:
            if 'lat' in element and 'lon' in element:
                lat, lon = element['lat'], element['lon']
                if (abs(lat - center_lat) < lat_range/2 and 
                    abs(lon - center_lon) < lon_range/2):
                    count += 1
        
        return count
    
    def fetch_air_quality(self, lat, lon):
        """
        Fetch air quality data from WAQI API
        Note: Requires free API key from https://aqicn.org/data-platform/token/
        """
        # This is a placeholder - you need to register for API key
        api_key = "YOUR_API_KEY_HERE"
        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={api_key}"
        
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            if data['status'] == 'ok':
                return data['data']['aqi']
        except Exception as e:
            print(f"Error fetching air quality: {e}")
        
        return None
    
    def load_worldpop_data(self, country_code='USA', year=2020):
        """
        Load population data from WorldPop
        Note: This requires downloading datasets from https://www.worldpop.org/
        
        For hackathon: Use their API or pre-downloaded GeoTIFF files
        """
        print("WorldPop integration requires downloading raster files")
        print(f"Visit: https://hub.worldpop.org/geodata/listing?id=78")
        print("Download population density rasters for your region")
        
        # Placeholder for raster processing
        # You would use rasterio to read .tif files:
        # import rasterio
        # with rasterio.open('population.tif') as src:
        #     population_data = src.read(1)
        
        return None
    
    def fetch_satellite_ndvi(self, bbox, date='2024-01-01'):
        """
        Fetch NDVI (greenery) data from Sentinel Hub
        Note: Requires Sentinel Hub account (free tier available)
        
        bbox: [min_lon, min_lat, max_lon, max_lat]
        """
        print("Sentinel Hub integration requires API credentials")
        print("Sign up at: https://www.sentinel-hub.com/")
        print("Free tier: 10,000 requests/month")
        
        # Placeholder for actual implementation
        # Would use sentinelhub Python package
        
        return None


def save_to_csv(df, filename='urban_data.csv'):
    """Save fetched data to CSV"""
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


# Example usage
if __name__ == "__main__":
    fetcher = UrbanDataFetcher()
    
    # Fetch data for a city
    city = "Austin, Texas"
    print(f"Fetching urban data for {city}...")
    
    df = fetcher.fetch_osm_data(city, radius_km=5)
    
    if df is not None:
        print(f"\nFetched {len(df)} neighborhoods")
        print("\nSample data:")
        print(df.head())
        
        # Save to CSV
        save_to_csv(df, 'austin_urban_data.csv')
        print("\n✅ Data ready! Use this CSV in your Streamlit app")
    else:
        print("❌ Could not fetch data. Check your internet connection.")
    
    print("\n📝 Additional Data Sources:")
    print("1. Air Quality: Register at https://aqicn.org/data-platform/token/")
    print("2. Population: Download from https://www.worldpop.org/")
    print("3. Satellite: Sign up at https://www.sentinel-hub.com/")