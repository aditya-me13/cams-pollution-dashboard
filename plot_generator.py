# plot_generator.py
# Generate air pollution maps for India

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web apps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pathlib import Path
import os
from datetime import datetime
from constants import INDIA_BOUNDS, COLOR_THEMES
import warnings
warnings.filterwarnings('ignore')


class IndiaMapPlotter:
    def __init__(self, plots_dir="plots"):
        """
        Initialize the map plotter
        
        Parameters:
        plots_dir (str): Directory to save plots
        """
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set matplotlib parameters for better plots
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['font.size'] = 10
        
    def create_india_map(self, data_values, metadata, color_theme=None, save_plot=True, custom_title=None):
        """
        Create air pollution map over India
        
        Parameters:
        data_values (np.array): 2D array of pollution data (lat, lon)
        metadata (dict): Metadata containing coordinates and variable info
        color_theme (str): Color map theme (default: from variable properties)
        save_plot (bool): Whether to save the plot
        custom_title (str): Custom title for the plot
        
        Returns:
        str: Path to saved plot file or None if not saved
        """
        try:
            # Extract metadata
            lats = metadata['lats']
            lons = metadata['lons']
            var_name = metadata['variable_name']
            display_name = metadata['display_name']
            units = metadata['units']
            pressure_level = metadata.get('pressure_level')
            time_stamp = metadata.get('timestamp_str')
            
            # Determine color theme
            if color_theme is None:
                # Use default from constants or fallback
                from constants import AIR_POLLUTION_VARIABLES
                if var_name in AIR_POLLUTION_VARIABLES:
                    color_theme = AIR_POLLUTION_VARIABLES[var_name]['cmap']
                else:
                    color_theme = 'viridis'
            
            # Validate color theme
            if color_theme not in COLOR_THEMES:
                print(f"Warning: Color theme '{color_theme}' not found, using 'viridis'")
                color_theme = 'viridis'
            
            # Create figure and axis
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            # Set map extent to India
            ax.set_extent([INDIA_BOUNDS['lon_min'], INDIA_BOUNDS['lon_max'], 
                            INDIA_BOUNDS['lat_min'], INDIA_BOUNDS['lat_max']], 
                            crs=ccrs.PlateCarree())
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.8, color='black')
            ax.add_feature(cfeature.LAND, alpha=0.2, color='lightgray')
            ax.add_feature(cfeature.OCEAN, alpha=0.2, color='lightblue')
            
            # Add states/provinces for better context
            ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5, alpha=0.5)
            
            # Create coordinate meshgrid
            if lons.ndim == 1 and lats.ndim == 1:
                lon_grid, lat_grid = np.meshgrid(lons, lats)
            else:
                lon_grid, lat_grid = lons, lats
            
            # Handle NaN values and calculate statistics
            valid_data = data_values[~np.isnan(data_values)]
            if len(valid_data) == 0:
                raise ValueError("All data values are NaN - cannot create plot")
            
            # Calculate percentile-based color limits
            from constants import AIR_POLLUTION_VARIABLES
            vmax_percentile = 90  # default
            if var_name in AIR_POLLUTION_VARIABLES:
                vmax_percentile = AIR_POLLUTION_VARIABLES[var_name]['vmax_percentile']
            
            vmin = np.nanpercentile(valid_data, 5)  # 5th percentile for minimum
            vmax = np.nanpercentile(valid_data, vmax_percentile)
            
            # Ensure vmin and vmax are different
            if vmax <= vmin:
                vmax = vmin + 0.1 * abs(vmin) if vmin != 0 else 1.0
            
            # Create contour plot
            levels = np.linspace(vmin, vmax, 25)
            contour = ax.contourf(lon_grid, lat_grid, data_values, 
                                levels=levels, cmap=color_theme, 
                                transform=ccrs.PlateCarree(), extend='max')
            
            # Add colorbar
            cbar = plt.colorbar(contour, ax=ax, shrink=0.6, pad=0.02, aspect=30)
            
            # Format colorbar label
            cbar_label = f"{display_name}"
            if units:
                cbar_label += f" ({units})"
            cbar.set_label(cbar_label, fontsize=12, labelpad=15)
            
            # Add gridlines
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                            linewidth=0.5, color='gray', alpha=0.7)
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size': 10}
            
            # Create title
            if custom_title:
                title = custom_title
            else:
                title = f'{display_name} Concentration over India'
                if pressure_level:
                    title += f' at {pressure_level} hPa'
                
                title += f' on {time_stamp}'
            
            plt.title(title, fontsize=14, pad=20, weight='bold')
            
            # Add statistics box
            stats_text = self._create_stats_text(valid_data, units)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
                    verticalalignment='top', fontsize=10)
            
            # Add color theme info
            theme_text = f"Color Theme: {COLOR_THEMES[color_theme]}"
            ax.text(0.98, 0.02, theme_text, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                    horizontalalignment='right', verticalalignment='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Save plot if requested
            plot_path = None
            if save_plot:
                plot_path = self._save_plot(fig, var_name, display_name, pressure_level, color_theme, time_stamp)
            
            plt.close(fig)  # Important: close figure to free memory
            
            return plot_path
            
        except Exception as e:
            plt.close('all')  # Clean up any open figures
            raise Exception(f"Error creating map: {str(e)}")
    
    def _create_stats_text(self, data, units):
        """Create statistics text for the plot"""
        units_str = f" {units}" if units else ""
        
        stats = {
            'Min': np.nanmin(data),
            'Max': np.nanmax(data),
            'Mean': np.nanmean(data),
            'Median': np.nanmedian(data),
            'Std': np.nanstd(data)
        }
        
        # Format numbers appropriately
        def format_number(val):
            if abs(val) >= 1000:
                return f"{val:.0f}"
            elif abs(val) >= 10:
                return f"{val:.1f}"
            else:
                return f"{val:.2f}"
        
        stats_lines = []
        for stat_name, stat_val in stats.items():
            stats_lines.append(f"{stat_name}: {format_number(stat_val)}{units_str}")
        
        return "\n".join(stats_lines)
    
    def _save_plot(self, fig, var_name, display_name, pressure_level, color_theme, time_stamp):
        """Save plot to file"""
        # Create filename
        safe_var_name = var_name.replace('/', '_').replace(' ', '_').replace('.', '_')
        safe_display_name = display_name.replace('/', '_').replace(' ', '_').replace('₂', '2').replace('₃', '3').replace('.', '_')
        safe_time_stamp = time_stamp.replace('-', '').replace(':', '').replace(' ', '_')
        
        filename_parts = [f"{safe_display_name}_India"]
        
        if pressure_level:
            filename_parts.append(f"{int(pressure_level)}hPa")
        
        filename_parts.extend([color_theme, safe_time_stamp])
        filename = "_".join(filename_parts) + ".png"
        
        plot_path = self.plots_dir / filename
        
        # Save with high DPI
        fig.savefig(plot_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        
        print(f"Plot saved: {plot_path}")
        return str(plot_path)
    
    def create_comparison_plot(self, datasets, titles, color_theme='viridis', save_plot=True):
        """
        Create comparison plot with multiple datasets
        
        Parameters:
        datasets (list): List of (data_values, metadata) tuples
        titles (list): List of titles for each subplot
        color_theme (str): Color map theme
        save_plot (bool): Whether to save the plot
        
        Returns:
        str: Path to saved plot file or None
        """
        try:
            n_datasets = len(datasets)
            if n_datasets < 2 or n_datasets > 4:
                raise ValueError("Comparison plot supports 2-4 datasets")
            
            # Determine subplot layout
            if n_datasets == 2:
                rows, cols = 1, 2
                figsize = (20, 8)
            elif n_datasets == 3:
                rows, cols = 1, 3
                figsize = (24, 7)
            else:  # 4 datasets
                rows, cols = 2, 2
                figsize = (16, 12)
            
            fig = plt.figure(figsize=figsize)
            
            # Find global min/max for consistent color scaling
            all_data = []
            for data_values, metadata in datasets:
                valid_data = data_values[~np.isnan(data_values)]
                all_data.extend(valid_data)
            
            global_min = np.percentile(all_data, 5)
            global_max = np.percentile(all_data, 95)
            
            plot_paths = []
            
            for i, (data_values, metadata) in enumerate(datasets):
                ax = fig.add_subplot(rows, cols, i+1, projection=ccrs.PlateCarree())
                
                # Set map extent to India
                ax.set_extent([INDIA_BOUNDS['lon_min'], INDIA_BOUNDS['lon_max'], 
                                INDIA_BOUNDS['lat_min'], INDIA_BOUNDS['lat_max']], 
                                crs=ccrs.PlateCarree())
                
                # Add map features
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
                ax.add_feature(cfeature.BORDERS, linewidth=0.6)
                ax.add_feature(cfeature.LAND, alpha=0.2, color='lightgray')
                
                # Plot data
                lats = metadata['lats']
                lons = metadata['lons']
                
                if lons.ndim == 1 and lats.ndim == 1:
                    lon_grid, lat_grid = np.meshgrid(lons, lats)
                else:
                    lon_grid, lat_grid = lons, lats
                
                levels = np.linspace(global_min, global_max, 20)
                contour = ax.contourf(lon_grid, lat_grid, data_values, 
                                    levels=levels, cmap=color_theme, 
                                    transform=ccrs.PlateCarree(), extend='max')
                
                # Add title
                ax.set_title(titles[i], fontsize=12, pad=10, weight='bold')
                
                # Add gridlines for first and last plots
                if i == 0 or i == len(datasets) - 1:
                    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.7)
                    gl.top_labels = False
                    gl.right_labels = False
            
            # Add colorbar
            units = datasets[0][1]['units']
            display_name = datasets[0][1]['display_name']
            cbar_label = f"{display_name}"
            if units:
                cbar_label += f" ({units})"
            
            cbar = fig.colorbar(contour, ax=fig.get_axes(), shrink=0.8, pad=0.02)
            cbar.set_label(cbar_label, fontsize=12)
            
            plt.tight_layout()
            
            # Save comparison plot
            if save_plot:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"comparison_{display_name.replace(' ', '_')}_{timestamp}.png"
                plot_path = self.plots_dir / filename
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plot_paths.append(str(plot_path))
            
            plt.close(fig)
            
            return plot_paths[0] if plot_paths else None
            
        except Exception as e:
            plt.close('all')
            raise Exception(f"Error creating comparison plot: {str(e)}")
    
    def list_available_themes(self):
        """Return available color themes"""
        return COLOR_THEMES
    
    def cleanup_old_plots(self, days_old=7):
        """
        Clean up plot files older than specified days
        
        Parameters:
        days_old (int): Delete plots older than this many days
        """
        try:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            deleted_count = 0
            for plot_file in self.plots_dir.glob("*.png"):
                if plot_file.stat().st_mtime < cutoff_date.timestamp():
                    plot_file.unlink()
                    deleted_count += 1
            
            print(f"Cleaned up {deleted_count} old plot files")
            return deleted_count
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            return 0


def test_plot_generator():
    """Test function for plot generator"""
    print("Testing plot generator...")
    
    # Create dummy data for testing
    lats = np.linspace(6, 38, 50)
    lons = np.linspace(68, 98, 60)
    
    # Create some sample pollution data
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    data = np.sin(lat_grid * 0.1) * np.cos(lon_grid * 0.1) * 100 + 50
    
    # Add some noise
    data += np.random.normal(0, 10, data.shape)
    
    metadata = {
        'variable_name': 'pm25',
        'display_name': 'PM2.5',
        'units': 'µg/m³',
        'lats': lats,
        'lons': lons,
        'pressure_level': None,
        'timestamp_str': '2023-10-01 12:00:00',
    }
    
    plotter = IndiaMapPlotter()
    
    try:
        plot_path = plotter.create_india_map(data, metadata, color_theme='YlOrRd')
        print(f"✅ Test plot created: {plot_path}")
        return True
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_plot_generator()