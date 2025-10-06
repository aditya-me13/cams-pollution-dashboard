# plot_generator.py
# Generate air pollution maps for India using GeoPandas for the map outline

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web apps
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from constants import INDIA_BOUNDS, COLOR_THEMES
import warnings
warnings.filterwarnings('ignore')


class IndiaMapPlotter:
    def __init__(self, plots_dir="plots", shapefile_path="shapefiles/India_State_Boundary.shp"):
        """
        Initialize the map plotter
        
        Parameters:
        plots_dir (str): Directory to save plots
        shapefile_path (str): Path to the India districts shapefile
        """
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        
        try:
            self.india_map = gpd.read_file(shapefile_path)

            # Ensure it's in lat/lon (WGS84)
            if self.india_map.crs is not None and self.india_map.crs.to_epsg() != 4326:
                self.india_map = self.india_map.to_crs(epsg=4326)

        except Exception as e:
            raise FileNotFoundError(f"Could not read the shapefile at '{shapefile_path}'. "
                                    f"Please ensure the file exists. Error: {e}")
        
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['font.size'] = 10
        
    def create_india_map(self, data_values, metadata, color_theme=None, save_plot=True, custom_title=None):
        """
        Create air pollution map over India
        """
        try:
            # Metadata extraction remains the same
            lats = metadata['lats']
            lons = metadata['lons']
            var_name = metadata['variable_name']
            display_name = metadata['display_name']
            units = metadata['units']
            pressure_level = metadata.get('pressure_level')
            time_stamp = metadata.get('timestamp_str')
            
            # Color theme logic remains the same
            if color_theme is None:
                from constants import AIR_POLLUTION_VARIABLES
                color_theme = AIR_POLLUTION_VARIABLES.get(var_name, {}).get('cmap', 'viridis')
            if color_theme not in COLOR_THEMES:
                print(f"Warning: Color theme '{color_theme}' not found, using 'viridis'")
                color_theme = 'viridis'
            
            # Create figure and axes
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(1, 1, 1)
            
            # Set map extent
            ax.set_xlim(INDIA_BOUNDS['lon_min'], INDIA_BOUNDS['lon_max'])
            ax.set_ylim(INDIA_BOUNDS['lat_min'], INDIA_BOUNDS['lat_max'])
            
            # --- KEY CHANGE: PLOT ORDER & ZORDER ---
            
            # 1. Plot the pollution data in the background (lower zorder)
            if lons.ndim == 1 and lats.ndim == 1:
                lon_grid, lat_grid = np.meshgrid(lons, lats)
            else:
                lon_grid, lat_grid = lons, lats
            
            valid_data = data_values[~np.isnan(data_values)]
            if len(valid_data) == 0:
                raise ValueError("All data values are NaN - cannot create plot")
            
            from constants import AIR_POLLUTION_VARIABLES
            vmax_percentile = AIR_POLLUTION_VARIABLES.get(var_name, {}).get('vmax_percentile', 90)
            vmin = np.nanpercentile(valid_data, 5)
            vmax = np.nanpercentile(valid_data, vmax_percentile)
            if vmax <= vmin:
                vmax = vmin + 1.0
            
            levels = np.linspace(vmin, vmax, 25)
            contour = ax.contourf(lon_grid, lat_grid, data_values, 
                                levels=levels, cmap=color_theme, extend='max', 
                                zorder=1) 

            # Auto-adjust bounds if INDIA_BOUNDS is too small or wrong
            xmin, ymin, xmax, ymax = self.india_map.total_bounds
            if not (INDIA_BOUNDS['lon_min'] <= xmin <= INDIA_BOUNDS['lon_max'] and INDIA_BOUNDS['lon_min'] <= xmax <= INDIA_BOUNDS['lon_max']):
                print("⚠️ Warning: Using shapefile's actual bounds instead of INDIA_BOUNDS.")
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

            # 2. Plot the India map outlines on top of the data (higher zorder)
            self.india_map.plot(ax=ax, edgecolor='black', facecolor='none', 
                                linewidth=0.8, zorder=2) # <-- CHANGED: Set zorder=2 (foreground)
            
            # Add colorbar
            cbar = plt.colorbar(contour, ax=ax, shrink=0.6, pad=0.02, aspect=30)
            cbar_label = f"{display_name}" + (f" ({units})" if units else "")
            cbar.set_label(cbar_label, fontsize=12, labelpad=15)
            
            # Add gridlines and labels
            ax.grid(True, linestyle='--', alpha=0.6, color='gray', zorder=3)
            ax.set_xlabel("Longitude", fontsize=10)
            ax.set_ylabel("Latitude", fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Title creation logic remains the same
            if custom_title:
                title = custom_title
            else:
                title = f'{display_name} Concentration over India'
                if pressure_level: title += f' at {pressure_level} hPa'
                title += f' on {time_stamp}'
            plt.title(title, fontsize=14, pad=20, weight='bold')
            
            # Statistics and theme info boxes remain the same
            stats_text = self._create_stats_text(valid_data, units)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
                    verticalalignment='top', fontsize=10, zorder=4)
            
            theme_text = f"Color Theme: {COLOR_THEMES[color_theme]}"
            ax.text(0.98, 0.02, theme_text, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                    horizontalalignment='right', verticalalignment='bottom', fontsize=9, zorder=4)
            
            plt.tight_layout()
            
            plot_path = None
            if save_plot:
                plot_path = self._save_plot(fig, var_name, display_name, pressure_level, color_theme, time_stamp)
            
            plt.close(fig)
            return plot_path
            
        except Exception as e:
            plt.close('all')
            raise Exception(f"Error creating map: {str(e)}")
    
    # All other helper methods (_create_stats_text, _save_plot, etc.) are unchanged.
    # The `create_comparison_plot` method is also left out for brevity but would need the same zorder fix.
    # The full, unchanged code for the helper methods from the previous answer is still valid.

    def _create_stats_text(self, data, units):
        units_str = f" {units}" if units else ""
        stats = {'Min': np.nanmin(data), 'Max': np.nanmax(data), 'Mean': np.nanmean(data), 'Median': np.nanmedian(data), 'Std': np.nanstd(data)}
        def format_number(val):
            if abs(val) >= 1000: return f"{val:.0f}"
            elif abs(val) >= 10: return f"{val:.1f}"
            else: return f"{val:.2f}"
        stats_lines = [f"{name}: {format_number(val)}{units_str}" for name, val in stats.items()]
        return "\n".join(stats_lines)

    def _save_plot(self, fig, var_name, display_name, pressure_level, color_theme, time_stamp):
        safe_display_name = display_name.replace('/', '_').replace(' ', '_').replace('₂', '2').replace('₃', '3').replace('.', '_')
        safe_time_stamp = time_stamp.replace('-', '').replace(':', '').replace(' ', '_')
        filename_parts = [f"{safe_display_name}_India"]
        if pressure_level:
            filename_parts.append(f"{int(pressure_level)}hPa")
        filename_parts.extend([color_theme, safe_time_stamp])
        filename = "_".join(filename_parts) + ".png"
        plot_path = self.plots_dir / filename
        fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Plot saved: {plot_path}")
        return str(plot_path)

    def list_available_themes(self):
        return COLOR_THEMES

def test_plot_generator():
    print("Testing plot generator with GeoPandas and zorder fix...")
    
    lats, lons = np.linspace(6, 38, 50), np.linspace(68, 98, 60)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    data = np.sin(lat_grid * 0.1) * np.cos(lon_grid * 0.1) * 100 + 50
    data += np.random.normal(0, 10, data.shape)
    
    metadata = {
        'variable_name': 'pm25', 'display_name': 'PM2.5', 'units': 'µg/m³',
        'lats': lats, 'lons': lons, 'pressure_level': None,
        'timestamp_str': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    shapefile_path = "shapefiles/India_State_Boundary.shp"
    if not Path(shapefile_path).exists():
        print(f"❌ Test failed: Shapefile not found at '{shapefile_path}'.")
        print("Please make sure you have unzipped 'India_State_Boundary.zip' into a 'shapefiles' folder.")
        return False
        
    plotter = IndiaMapPlotter(shapefile_path=shapefile_path)
    
    try:
        plot_path = plotter.create_india_map(data, metadata, color_theme='YlOrRd')
        print(f"✅ Test plot created successfully: {plot_path}")
        return True
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_plot_generator()