# Air pollution variables and their properties

AIR_POLLUTION_VARIABLES = {
    # PM2.5
    'pm2p5': {'units': 'µg/m³', 'name': 'PM2.5', 'cmap': 'YlOrRd', 'vmax_percentile': 95, 'type': 'surface'},
    'pm25': {'units': 'µg/m³', 'name': 'PM2.5', 'cmap': 'YlOrRd', 'vmax_percentile': 95, 'type': 'surface'},
    'PM2P5': {'units': 'µg/m³', 'name': 'PM2.5', 'cmap': 'YlOrRd', 'vmax_percentile': 95, 'type': 'surface'},
    'PM25': {'units': 'µg/m³', 'name': 'PM2.5', 'cmap': 'YlOrRd', 'vmax_percentile': 95, 'type': 'surface'},
    'pm2_5': {'units': 'µg/m³', 'name': 'PM2.5', 'cmap': 'YlOrRd', 'vmax_percentile': 95, 'type': 'surface'},
    'PM2_5': {'units': 'µg/m³', 'name': 'PM2.5', 'cmap': 'YlOrRd', 'vmax_percentile': 95, 'type': 'surface'},
    'particulate_matter_2_5': {'units': 'µg/m³', 'name': 'PM2.5', 'cmap': 'YlOrRd', 'vmax_percentile': 95, 'type': 'surface'},
    'particulate_matter_2.5um': {'units': 'µg/m³', 'name': 'PM2.5', 'cmap': 'YlOrRd', 'vmax_percentile': 95, 'type': 'surface'},
    'mass_concentration_of_pm2p5_ambient_aerosol_particles_in_air': {'units': 'µg/m³', 'name': 'PM2.5', 'cmap': 'YlOrRd', 'vmax_percentile': 95, 'type': 'surface'},
    
    # PM10
    'pm10': {'units': 'µg/m³', 'name': 'PM10', 'cmap': 'Oranges', 'vmax_percentile': 95, 'type': 'surface'},
    'PM10': {'units': 'µg/m³', 'name': 'PM10', 'cmap': 'Oranges', 'vmax_percentile': 95, 'type': 'surface'},
    'pm10p0': {'units': 'µg/m³', 'name': 'PM10', 'cmap': 'Oranges', 'vmax_percentile': 95, 'type': 'surface'},
    'particulate_matter_10': {'units': 'µg/m³', 'name': 'PM10', 'cmap': 'Oranges', 'vmax_percentile': 95, 'type': 'surface'},
    'particulate_matter_10um': {'units': 'µg/m³', 'name': 'PM10', 'cmap': 'Oranges', 'vmax_percentile': 95, 'type': 'surface'},
    'mass_concentration_of_pm10_ambient_aerosol_particles_in_air': {'units': 'µg/m³', 'name': 'PM10', 'cmap': 'Oranges', 'vmax_percentile': 95, 'type': 'surface'},
    
    # PM1
    'pm1': {'units': 'µg/m³', 'name': 'PM1', 'cmap': 'Reds', 'vmax_percentile': 95, 'type': 'surface'},
    'PM1': {'units': 'µg/m³', 'name': 'PM1', 'cmap': 'Reds', 'vmax_percentile': 95, 'type': 'surface'},
    'particulate_matter_1um': {'units': 'µg/m³', 'name': 'PM1', 'cmap': 'Reds', 'vmax_percentile': 95, 'type': 'surface'},
    
    # NO2
    'no2': {'units': 'µg/m³', 'name': 'NO₂', 'cmap': 'Reds', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'NO2': {'units': 'µg/m³', 'name': 'NO₂', 'cmap': 'Reds', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'nitrogen_dioxide': {'units': 'µg/m³', 'name': 'NO₂', 'cmap': 'Reds', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'mass_concentration_of_nitrogen_dioxide_in_air': {'units': 'µg/m³', 'name': 'NO₂', 'cmap': 'Reds', 'vmax_percentile': 90, 'type': 'atmospheric'},
    
    # SO2
    'so2': {'units': 'µg/m³', 'name': 'SO₂', 'cmap': 'Purples', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'SO2': {'units': 'µg/m³', 'name': 'SO₂', 'cmap': 'Purples', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'sulphur_dioxide': {'units': 'µg/m³', 'name': 'SO₂', 'cmap': 'Purples', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'sulfur_dioxide': {'units': 'µg/m³', 'name': 'SO₂', 'cmap': 'Purples', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'mass_concentration_of_sulfur_dioxide_in_air': {'units': 'µg/m³', 'name': 'SO₂', 'cmap': 'Purples', 'vmax_percentile': 90, 'type': 'atmospheric'},
    
    # O3
    'o3': {'units': 'µg/m³', 'name': 'O₃', 'cmap': 'Blues', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'O3': {'units': 'µg/m³', 'name': 'O₃', 'cmap': 'Blues', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'ozone': {'units': 'µg/m³', 'name': 'O₃', 'cmap': 'Blues', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'mass_concentration_of_ozone_in_air': {'units': 'µg/m³', 'name': 'O₃', 'cmap': 'Blues', 'vmax_percentile': 90, 'type': 'atmospheric'},
    
    # CO
    'co': {'units': 'mg/m³', 'name': 'CO', 'cmap': 'Greens', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'CO': {'units': 'mg/m³', 'name': 'CO', 'cmap': 'Greens', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'carbon_monoxide': {'units': 'mg/m³', 'name': 'CO', 'cmap': 'Greens', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'mass_concentration_of_carbon_monoxide_in_air': {'units': 'mg/m³', 'name': 'CO', 'cmap': 'Greens', 'vmax_percentile': 90, 'type': 'atmospheric'},
    
    # NO (Nitrogen Monoxide)
    'no': {'units': 'µg/m³', 'name': 'NO', 'cmap': 'Oranges', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'NO': {'units': 'µg/m³', 'name': 'NO', 'cmap': 'Oranges', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'nitrogen_monoxide': {'units': 'µg/m³', 'name': 'NO', 'cmap': 'Oranges', 'vmax_percentile': 90, 'type': 'atmospheric'},
    
    # NH3
    'nh3': {'units': 'µg/m³', 'name': 'NH₃', 'cmap': 'viridis', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'NH3': {'units': 'µg/m³', 'name': 'NH₃', 'cmap': 'viridis', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'ammonia': {'units': 'µg/m³', 'name': 'NH₃', 'cmap': 'viridis', 'vmax_percentile': 90, 'type': 'atmospheric'},
    'mass_concentration_of_ammonia_in_air': {'units': 'µg/m³', 'name': 'NH₃', 'cmap': 'viridis', 'vmax_percentile': 90, 'type': 'atmospheric'},
    
    # Total Column variables (these are surface-level total column measurements)
    'total_column_carbon_monoxide': {'units': 'mol/m²', 'name': 'Total Column CO', 'cmap': 'Greens', 'vmax_percentile': 90, 'type': 'surface'},
    'total_column_nitrogen_monoxide': {'units': 'mol/m²', 'name': 'Total Column NO', 'cmap': 'Oranges', 'vmax_percentile': 90, 'type': 'surface'},
    'total_column_nitrogen_dioxide': {'units': 'mol/m²', 'name': 'Total Column NO₂', 'cmap': 'Reds', 'vmax_percentile': 90, 'type': 'surface'},
    'total_column_ozone': {'units': 'mol/m²', 'name': 'Total Column O₃', 'cmap': 'Blues', 'vmax_percentile': 90, 'type': 'surface'},
    'total_column_sulphur_dioxide': {'units': 'mol/m²', 'name': 'Total Column SO₂', 'cmap': 'Purples', 'vmax_percentile': 90, 'type': 'surface'},
    
    # Legacy total column names
    'tcno2': {'units': 'mol/m²', 'name': 'Total Column NO₂', 'cmap': 'Reds', 'vmax_percentile': 90, 'type': 'surface'},
    'tcso2': {'units': 'mol/m²', 'name': 'Total Column SO₂', 'cmap': 'Purples', 'vmax_percentile': 90, 'type': 'surface'},
    'tco3': {'units': 'mol/m²', 'name': 'Total Column O₃', 'cmap': 'Blues', 'vmax_percentile': 90, 'type': 'surface'},
    'tcco': {'units': 'mol/m²', 'name': 'Total Column CO', 'cmap': 'Greens', 'vmax_percentile': 90, 'type': 'surface'},
    
    # AOD (Aerosol Optical Depth) - surface measurement
    'aod550': {'units': '', 'name': 'AOD 550nm', 'cmap': 'plasma', 'vmax_percentile': 95, 'type': 'surface'},
    'aod': {'units': '', 'name': 'AOD', 'cmap': 'plasma', 'vmax_percentile': 95, 'type': 'surface'},
    'aerosol_optical_depth': {'units': '', 'name': 'AOD', 'cmap': 'plasma', 'vmax_percentile': 95, 'type': 'surface'},
}

# Available color themes for plotting
COLOR_THEMES = {
    'YlOrRd': 'Yellow-Orange-Red',
    'Oranges': 'Oranges',
    'Reds': 'Reds', 
    'Purples': 'Purples',
    'Blues': 'Blues',
    'Greens': 'Greens',
    'viridis': 'Viridis',
    'plasma': 'Plasma',
    'inferno': 'Inferno',
    'magma': 'Magma',
    'cividis': 'Cividis',
    'coolwarm': 'Cool-Warm',
    'RdYlBu': 'Red-Yellow-Blue',
    'Spectral': 'Spectral'
}

# India map boundaries
INDIA_BOUNDS = {
    'lat_min': 6.0,
    'lat_max': 38.0,
    'lon_min': 68.0,
    'lon_max': 98.0
}

# Common pressure levels for atmospheric variables (in hPa)
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# File upload settings
ALLOWED_EXTENSIONS = {'nc', 'zip'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB