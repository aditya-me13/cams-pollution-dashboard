# data_processor.py
# NetCDF file processing and air pollution variable detection

import numpy as np
import pandas as pd
import xarray as xr
import zipfile
import os
from pathlib import Path
from constants import AIR_POLLUTION_VARIABLES, PRESSURE_LEVELS
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class NetCDFProcessor:
    def __init__(self, file_path):
        """
        Initialize NetCDF processor
        
        Parameters:
        file_path (str): Path to NetCDF or ZIP file
        """
        self.file_path = Path(file_path)
        self.dataset = None
        self.surface_dataset = None
        self.atmospheric_dataset = None
        self.detected_variables = {}
        
    def load_dataset(self):
        """Load NetCDF dataset from file or ZIP"""
        try:
            if self.file_path.suffix.lower() == '.zip':
                return self._load_from_zip()
            elif self.file_path.suffix.lower() == '.nc':
                return self._load_from_netcdf()
            else:
                raise ValueError("Unsupported file format. Use .nc or .zip files.")
                
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def _load_from_zip(self):
        """Load dataset from ZIP file (CAMS format)"""
        with zipfile.ZipFile(self.file_path, 'r') as zf:
            zip_contents = zf.namelist()
            
            # Look for surface and atmospheric data files
            surface_file = None
            atmospheric_file = None
            
            for file in zip_contents:
                if 'sfc' in file.lower() or 'surface' in file.lower():
                    surface_file = file
                elif 'plev' in file.lower() or 'pressure' in file.lower() or 'atmospheric' in file.lower():
                    atmospheric_file = file
            
            # Load surface data if available
            if surface_file:
                with zf.open(surface_file) as f:
                    self.surface_dataset = xr.open_dataset(f, engine='netcdf4')
                    print(f"Loaded surface data: {surface_file}")
            
            # Load atmospheric data if available
            if atmospheric_file:
                with zf.open(atmospheric_file) as f:
                    self.atmospheric_dataset = xr.open_dataset(f, engine='netcdf4')
                    print(f"Loaded atmospheric data: {atmospheric_file}")
            
            # If no specific files found, try to load the first .nc file
            if not surface_file and not atmospheric_file:
                nc_files = [f for f in zip_contents if f.endswith('.nc')]
                if nc_files:
                    with zf.open(nc_files[0]) as f:
                        self.dataset = xr.open_dataset(f, engine='netcdf4')
                        print(f"Loaded dataset: {nc_files[0]}")
                else:
                    raise ValueError("No NetCDF files found in ZIP")
        
        return True
    
    def _load_from_netcdf(self):
        """Load dataset from single NetCDF file"""
        self.dataset = xr.open_dataset(self.file_path)
        print(f"Loaded NetCDF file: {self.file_path.name}")
        return True
    
    def detect_variables(self):
        """Detect air pollution variables in all loaded datasets"""
        self.detected_variables = {}
        
        # Check surface dataset
        if self.surface_dataset is not None:
            surface_vars = self._detect_variables_in_dataset(self.surface_dataset, 'surface')
            self.detected_variables.update(surface_vars)
        
        # Check atmospheric dataset
        if self.atmospheric_dataset is not None:
            atmo_vars = self._detect_variables_in_dataset(self.atmospheric_dataset, 'atmospheric')
            self.detected_variables.update(atmo_vars)
        
        # Check main dataset if no separate files
        if self.dataset is not None:
            main_vars = self._detect_variables_in_dataset(self.dataset, 'unknown')
            self.detected_variables.update(main_vars)
        
        return self.detected_variables
    
    def _detect_variables_in_dataset(self, dataset, dataset_type):
        """Detect air pollution variables in a specific dataset"""
        detected = {}
        
        for var_name in dataset.data_vars:
            var_name_lower = var_name.lower()
            
            # Check exact matches first
            if var_name in AIR_POLLUTION_VARIABLES:
                detected[var_name] = AIR_POLLUTION_VARIABLES[var_name].copy()
                detected[var_name]['original_name'] = var_name
                detected[var_name]['dataset_type'] = dataset_type
                detected[var_name]['shape'] = dataset[var_name].shape
                detected[var_name]['dims'] = list(dataset[var_name].dims)
                
            elif var_name_lower in AIR_POLLUTION_VARIABLES:
                detected[var_name] = AIR_POLLUTION_VARIABLES[var_name_lower].copy()
                detected[var_name]['original_name'] = var_name
                detected[var_name]['dataset_type'] = dataset_type
                detected[var_name]['shape'] = dataset[var_name].shape
                detected[var_name]['dims'] = list(dataset[var_name].dims)
                
            else:
                # Check for partial matches
                var_info = dataset[var_name]
                long_name = getattr(var_info, 'long_name', '').lower()
                standard_name = getattr(var_info, 'standard_name', '').lower()
                
                # Check for keywords
                pollution_keywords = {
                    'pm2.5': {'units': 'µg/m³', 'name': 'PM2.5', 'cmap': 'YlOrRd', 'vmax_percentile': 95, 'type': 'surface'},
                    'pm10': {'units': 'µg/m³', 'name': 'PM10', 'cmap': 'Oranges', 'vmax_percentile': 95, 'type': 'surface'},
                    'pm1': {'units': 'µg/m³', 'name': 'PM1', 'cmap': 'Reds', 'vmax_percentile': 95, 'type': 'surface'},
                    'no2': {'units': 'µg/m³', 'name': 'NO₂', 'cmap': 'Reds', 'vmax_percentile': 90, 'type': 'atmospheric'},
                    'nitrogen dioxide': {'units': 'µg/m³', 'name': 'NO₂', 'cmap': 'Reds', 'vmax_percentile': 90, 'type': 'atmospheric'},
                    'so2': {'units': 'µg/m³', 'name': 'SO₂', 'cmap': 'Purples', 'vmax_percentile': 90, 'type': 'atmospheric'},
                    'sulphur dioxide': {'units': 'µg/m³', 'name': 'SO₂', 'cmap': 'Purples', 'vmax_percentile': 90, 'type': 'atmospheric'},
                    'sulfur dioxide': {'units': 'µg/m³', 'name': 'SO₂', 'cmap': 'Purples', 'vmax_percentile': 90, 'type': 'atmospheric'},
                    'ozone': {'units': 'µg/m³', 'name': 'O₃', 'cmap': 'Blues', 'vmax_percentile': 90, 'type': 'atmospheric'},
                    'carbon monoxide': {'units': 'mg/m³', 'name': 'CO', 'cmap': 'Greens', 'vmax_percentile': 90, 'type': 'atmospheric'},
                    'nitrogen monoxide': {'units': 'µg/m³', 'name': 'NO', 'cmap': 'Oranges', 'vmax_percentile': 90, 'type': 'atmospheric'},
                    'ammonia': {'units': 'µg/m³', 'name': 'NH₃', 'cmap': 'viridis', 'vmax_percentile': 90, 'type': 'atmospheric'},
                    'particulate': {'units': 'µg/m³', 'name': 'Particulate Matter', 'cmap': 'YlOrRd', 'vmax_percentile': 95, 'type': 'surface'},
                }
                
                for keyword, properties in pollution_keywords.items():
                    if (keyword in var_name_lower or 
                        keyword in long_name or 
                        keyword in standard_name):
                        detected[var_name] = properties.copy()
                        detected[var_name]['original_name'] = var_name
                        detected[var_name]['dataset_type'] = dataset_type
                        detected[var_name]['shape'] = dataset[var_name].shape
                        detected[var_name]['dims'] = list(dataset[var_name].dims)
                        break
        
        return detected
    
    def get_coordinates(self, dataset):
        """Get coordinate names from dataset"""
        coords = list(dataset.coords.keys())
        
        # Find latitude coordinate
        lat_names = ['latitude', 'lat', 'y', 'Latitude', 'LATITUDE']
        lat_coord = next((name for name in lat_names if name in coords), None)
        
        # Find longitude coordinate
        lon_names = ['longitude', 'lon', 'x', 'Longitude', 'LONGITUDE']
        lon_coord = next((name for name in lon_names if name in coords), None)
        
        # Find time coordinate
        time_names = ['time', 'Time', 'TIME', 'forecast_reference_time']
        time_coord = next((name for name in time_names if name in coords), None)
        
        # Find pressure/level coordinate
        level_names = ['pressure_level', 'plev', 'level', 'pressure', 'lev']
        level_coord = next((name for name in level_names if name in coords), None)
        
        return {
            'lat': lat_coord,
            'lon': lon_coord,
            'time': time_coord,
            'level': level_coord
        }
    
    def format_timestamp(self, timestamp):
        """Format timestamp for display in plots"""
        try:
            if pd.isna(timestamp):
                return "Unknown Time"
            
            # Convert to pandas datetime if it isn't already
            if not isinstance(timestamp, pd.Timestamp):
                timestamp = pd.to_datetime(timestamp)
            
            # Format as "YYYY-MM-DD HH:MM"
            return timestamp.strftime('%Y-%m-%d %H:%M')
        except:
            return str(timestamp)
    
    def extract_data(self, variable_name, time_index=1, pressure_level=None):
        """
        Extract data for a specific variable
        
        Parameters:
        variable_name (str): Name of the variable to extract
        time_index (int): Time index to extract (default: 0 for current time)
        pressure_level (float): Pressure level for atmospheric variables (default: surface level)
        
        Returns:
        tuple: (data_array, metadata)
        """
        if variable_name not in self.detected_variables:
            raise ValueError(f"Variable {variable_name} not found in detected variables")
        
        var_info = self.detected_variables[variable_name]
        dataset_type = var_info['dataset_type']
        
        # Determine which dataset to use
        if dataset_type == 'surface' and self.surface_dataset is not None:
            dataset = self.surface_dataset
        elif dataset_type == 'atmospheric' and self.atmospheric_dataset is not None:
            dataset = self.atmospheric_dataset
        elif self.dataset is not None:
            dataset = self.dataset
        else:
            raise ValueError(f"No suitable dataset found for variable {variable_name}")
        
        # Get the data variable
        data_var = dataset[variable_name]
        coords = self.get_coordinates(dataset)
        print(f"Coordinates: {coords}\n\n")

        # Handle different data shapes
        data_array = data_var
        print(f"Data array shape: {data_array.dims} \n\n")

        # Get timestamp information before extracting data
        selected_timestamp = None
        timestamp_str = "Unknown Time"
        
        # Handle time dimension
        if coords['time'] and coords['time'] in data_array.dims:
            # Get all available times
            available_times = pd.to_datetime(dataset[coords['time']].values)
            
            if time_index == -1:  # Latest time
                time_index = len(available_times) - 1
            
            # Ensure time_index is within bounds
            if 0 <= time_index < len(available_times):
                selected_timestamp = available_times[time_index]
                timestamp_str = self.format_timestamp(selected_timestamp)
                print(f"Time index: {time_index} selected - {timestamp_str}")
                data_array = data_array.isel({coords['time']: time_index})
            else:
                print(f"Warning: time_index {time_index} out of bounds, using index 0")
                time_index = 0
                selected_timestamp = available_times[time_index]
                timestamp_str = self.format_timestamp(selected_timestamp)
                data_array = data_array.isel({coords['time']: time_index})
        
        # Handle pressure/level dimension for atmospheric variables
        if coords['level'] and coords['level'] in data_array.dims:
            if pressure_level is None:
                # Default to surface level (highest pressure)
                pressure_level = 1000
            
            # Find closest pressure level
            pressure_values = dataset[coords['level']].values
            level_index = np.argmin(np.abs(pressure_values - pressure_level))
            actual_pressure = pressure_values[level_index]
            
            data_array = data_array.isel({coords['level']: level_index})
            print(f"Selected pressure level: {actual_pressure} hPa (requested: {pressure_level} hPa)")
        
        # Handle batch dimension (usually the first dimension for CAMS data)
        shape = data_array.shape
        if len(shape) == 4:  # (batch, time, lat, lon) or similar
            data_array = data_array[0, -1]  # Take first batch, latest time
        elif len(shape) == 3:  # (batch, lat, lon) or (time, lat, lon)
            data_array = data_array[-1]  # Take latest
        elif len(shape) == 5:  # (batch, time, level, lat, lon)
            data_array = data_array[0, -1]  # Already handled level above
        
        # Get coordinate arrays
        lats = dataset[coords['lat']].values
        lons = dataset[coords['lon']].values
        
        # Convert units if necessary
        original_units = getattr(dataset[variable_name], 'units', '')
        data_values = self._convert_units(data_array.values, original_units, var_info['units'])

        metadata = {
            'variable_name': variable_name,
            'display_name': var_info['name'],
            'units': var_info['units'],
            'original_units': original_units,
            'shape': data_values.shape,
            'lats': lats,
            'lons': lons,
            'pressure_level': pressure_level if coords['level'] and coords['level'] in dataset[variable_name].dims else None,
            'time_index': time_index,
            'timestamp': selected_timestamp,
            'timestamp_str': timestamp_str,
            'dataset_type': dataset_type
        }
        
        return data_values, metadata
    
    def _convert_units(self, data, original_units, target_units):
        """Convert data units for air pollution variables"""
        data_converted = data.copy()
        
        if original_units and target_units:
            orig_lower = original_units.lower()
            target_lower = target_units.lower()
            
            # kg/m³ to µg/m³
            if 'kg' in orig_lower and 'µg' in target_lower:
                data_converted = data_converted * 1e9
                print(f"Converting from {original_units} to {target_units} (×1e9)")
            
            # kg/m³ to mg/m³
            elif 'kg' in orig_lower and 'mg' in target_lower:
                data_converted = data_converted * 1e6
                print(f"Converting from {original_units} to {target_units} (×1e6)")
            
            # mol/m² conversions (keep as is)
            elif 'mol' in orig_lower:
                print(f"Units {original_units} kept as is")
            
            # No unit (dimensionless) - keep as is
            elif target_units == '':
                print("Dimensionless variable - no unit conversion")
        
        return data_converted
    
    def get_available_times(self, variable_name):
        """Get available time steps for a variable"""
        if variable_name not in self.detected_variables:
            return []
        
        var_info = self.detected_variables[variable_name]
        dataset_type = var_info['dataset_type']
        
        # Determine which dataset to use
        if dataset_type == 'surface' and self.surface_dataset is not None:
            dataset = self.surface_dataset
        elif dataset_type == 'atmospheric' and self.atmospheric_dataset is not None:
            dataset = self.atmospheric_dataset
        elif self.dataset is not None:
            dataset = self.dataset
        else:
            return []
        
        coords = self.get_coordinates(dataset)
        
        if coords['time'] and coords['time'] in dataset.dims:
            times = pd.to_datetime(dataset[coords['time']].values)
            print(f"Times: {times.to_list()}")
            return times.tolist()
        
        return []
    
    def get_available_pressure_levels(self, variable_name):
        """Get available pressure levels for atmospheric variables"""
        if variable_name not in self.detected_variables:
            return []
        
        var_info = self.detected_variables[variable_name]
        if var_info['type'] != 'atmospheric':
            return []
        
        dataset_type = var_info['dataset_type']
        
        # Determine which dataset to use
        if dataset_type == 'atmospheric' and self.atmospheric_dataset is not None:
            dataset = self.atmospheric_dataset
        elif self.dataset is not None:
            dataset = self.dataset
        else:
            return []
        
        coords = self.get_coordinates(dataset)
        
        if coords['level'] and coords['level'] in dataset.dims:
            levels = dataset[coords['level']].values
            return levels.tolist()
        
        return PRESSURE_LEVELS  # Default pressure levels
    
    def close(self):
        """Close all open datasets"""
        if self.dataset is not None:
            self.dataset.close()
        if self.surface_dataset is not None:
            self.surface_dataset.close()
        if self.atmospheric_dataset is not None:
            self.atmospheric_dataset.close()


def analyze_netcdf_file(file_path):
    """
    Analyze NetCDF file structure and return summary
    
    Parameters:
    file_path (str): Path to NetCDF or ZIP file
    
    Returns:
    dict: Analysis summary
    """
    processor = NetCDFProcessor(file_path)
    
    try:
        processor.load_dataset()
        detected_vars = processor.detect_variables()
        
        analysis = {
            'success': True,
            'file_path': str(file_path),
            'detected_variables': detected_vars,
            'total_variables': len(detected_vars),
            'surface_variables': len([v for v in detected_vars.values() if v.get('type') == 'surface']),
            'atmospheric_variables': len([v for v in detected_vars.values() if v.get('type') == 'atmospheric']),
        }
        
        # Get sample time information
        if detected_vars:
            sample_var = list(detected_vars.keys())[0]
            times = processor.get_available_times(sample_var)
            if times:
                analysis['time_range'] = {
                    'start': str(times[0]),
                    'end': str(times[-1]),
                    'count': len(times)
                }
        
        processor.close()
        return analysis
        
    except Exception as e:
        processor.close()
        return {
            'success': False,
            'error': str(e),
            'file_path': str(file_path)
        }