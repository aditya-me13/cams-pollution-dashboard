# cams_downloader.py
# Download CAMS atmospheric composition data

import cdsapi
import zipfile
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

class CAMSDownloader:
    def __init__(self, download_dir="downloads"):
        """
        Initialize CAMS downloader
        
        Parameters:
        download_dir (str): Directory to store downloaded files
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.extracted_dir = self.download_dir / "extracted"
        self.extracted_dir.mkdir(exist_ok=True)
        
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize CDS API client"""
        try:
            # Try to read .cdsapirc file from current directory first, then home directory
            cdsapirc_path = Path.cwd() / ".cdsapirc"
            if not cdsapirc_path.exists():
                cdsapirc_path = Path.home() / ".cdsapirc"
            
            if cdsapirc_path.exists():
                # Parse credentials from .cdsapirc
                with open(cdsapirc_path, 'r') as f:
                    lines = f.readlines()
                
                url = None
                key = None
                for line in lines:
                    line = line.strip()
                    if line.startswith('url:'):
                        url = line.split(':', 1)[1].strip()
                    elif line.startswith('key:'):
                        key = line.split(':', 1)[1].strip()
                
                print(url, key)
                if url and key:
                    # Initialize with timeout to prevent hanging
                    import signal
                    def timeout_handler(signum, frame):
                        raise TimeoutError("CDS API initialization timeout")

                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(5)  # 5-second timeout

                    try:
                        self.client = cdsapi.Client(key=key, url=url)
                        signal.alarm(0)  # Cancel the alarm
                        print("✅ CDS API client initialized from .cdsapirc")
                    except TimeoutError:
                        signal.alarm(0)
                        raise TimeoutError("CDS API client initialization timed out")
                else:
                    raise ValueError("Could not parse URL or key from .cdsapirc file")
            else:
                # Try default initialization (will look for environment variables)
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("CDS API initialization timeout")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)  # 5-second timeout

                try:
                    self.client = cdsapi.Client()
                    signal.alarm(0)  # Cancel the alarm
                    print("✅ CDS API client initialized with default settings")
                except TimeoutError:
                    signal.alarm(0)
                    raise TimeoutError("CDS API client initialization timed out")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize CDS API client: {str(e)}")
            print("Please ensure you have:")
            print("1. Created an account at https://cds.climate.copernicus.eu/")
            print("2. Created a .cdsapirc file in your home directory with your credentials")
            print("3. Or set CDSAPI_URL and CDSAPI_KEY environment variables")
            self.client = None
    
    def is_client_ready(self):
        """Check if CDS API client is ready"""
        return self.client is not None
    
    def download_cams_data(self, date_str, variables=None, pressure_levels=None):
        """
        Download CAMS atmospheric composition data for a specific date
        
        Parameters:
        date_str (str): Date in YYYY-MM-DD format
        variables (list): List of variables to download (default: common air pollution variables)
        pressure_levels (list): List of pressure levels (default: standard levels)
        
        Returns:
        str: Path to downloaded ZIP file
        """
        if not self.is_client_ready():
            raise Exception("CDS API client not initialized. Please check your credentials.")
        
        # Validate date
        try:
            target_date = pd.to_datetime(date_str)
            date_str = target_date.strftime('%Y-%m-%d')
        except:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD format.")
        
        # Check if data already exists
        filename = f"{date_str}-cams.nc.zip"
        filepath = self.download_dir / filename
        
        if filepath.exists():
            print(f"✅ Data for {date_str} already exists: {filename}")
            return str(filepath)
        
        # Default variables (common air pollution variables)
        if variables is None:
            variables = [
                # Meteorological surface-level variables
                "10m_u_component_of_wind",
                "10m_v_component_of_wind", 
                "2m_temperature",
                "mean_sea_level_pressure",
                
                # Pollution surface-level variables
                "particulate_matter_1um",
                "particulate_matter_2.5um", 
                "particulate_matter_10um",
                "total_column_carbon_monoxide",
                "total_column_nitrogen_monoxide",
                "total_column_nitrogen_dioxide",
                "total_column_ozone",
                "total_column_sulphur_dioxide",
                
                # Meteorological atmospheric variables
                "u_component_of_wind",
                "v_component_of_wind",
                "temperature", 
                "geopotential",
                "specific_humidity",
                
                # Pollution atmospheric variables
                "carbon_monoxide",
                "nitrogen_dioxide",
                "nitrogen_monoxide", 
                "ozone",
                "sulphur_dioxide",
            ]
        
        # Default pressure levels
        if pressure_levels is None:
            pressure_levels = [
                "50", "100", "150", "200", "250", "300", "400",
                "500", "600", "700", "850", "925", "1000",
            ]
        
        print(f"🔄 Downloading CAMS data for {date_str}...")
        print(f"Variables: {len(variables)} selected")
        print(f"Pressure levels: {len(pressure_levels)} levels")
        
        try:
            # Make the API request
            self.client.retrieve(
                "cams-global-atmospheric-composition-forecasts",
                {
                    "type": "forecast",
                    "leadtime_hour": "0",
                    "variable": variables,
                    "pressure_level": pressure_levels,
                    "date": date_str,
                    "time": ["00:00", "12:00"],  # Two time steps
                    "format": "netcdf_zip",
                },
                str(filepath),
            )
            
            print(f"✅ Successfully downloaded: {filename}")
            return str(filepath)
            
        except Exception as e:
            # Clean up partial download
            if filepath.exists():
                filepath.unlink()
            raise Exception(f"Error downloading CAMS data: {str(e)}")
    
    def extract_cams_files(self, zip_path):
        """
        Extract surface and atmospheric data from CAMS ZIP file
        
        Parameters:
        zip_path (str): Path to CAMS ZIP file
        
        Returns:
        dict: Paths to extracted files
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")
        
        # Extract date from filename
        date_str = zip_path.stem.replace("-cams.nc", "")
        
        surface_path = self.extracted_dir / f"{date_str}-cams-surface.nc"
        atmospheric_path = self.extracted_dir / f"{date_str}-cams-atmospheric.nc"
        
        extracted_files = {}
        
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zip_contents = zf.namelist()
                
                # Extract surface data
                surface_file = None
                for file in zip_contents:
                    if 'sfc' in file.lower() or file.endswith('_sfc.nc'):
                        surface_file = file
                        break
                
                if surface_file and not surface_path.exists():
                    with open(surface_path, "wb") as f:
                        f.write(zf.read(surface_file))
                    print(f"✅ Extracted surface data: {surface_path.name}")
                    extracted_files['surface'] = str(surface_path)
                elif surface_path.exists():
                    extracted_files['surface'] = str(surface_path)
                
                # Extract atmospheric data  
                atmospheric_file = None
                for file in zip_contents:
                    if 'plev' in file.lower() or file.endswith('_plev.nc'):
                        atmospheric_file = file
                        break
                
                if atmospheric_file and not atmospheric_path.exists():
                    with open(atmospheric_path, "wb") as f:
                        f.write(zf.read(atmospheric_file))
                    print(f"✅ Extracted atmospheric data: {atmospheric_path.name}")
                    extracted_files['atmospheric'] = str(atmospheric_path)
                elif atmospheric_path.exists():
                    extracted_files['atmospheric'] = str(atmospheric_path)
                
                # If no specific files found, extract all .nc files
                if not extracted_files:
                    nc_files = [f for f in zip_contents if f.endswith('.nc')]
                    for nc_file in nc_files:
                        output_path = self.extracted_dir / nc_file
                        if not output_path.exists():
                            with open(output_path, "wb") as f:
                                f.write(zf.read(nc_file))
                            extracted_files[nc_file] = str(output_path)
        
        except Exception as e:
            raise Exception(f"Error extracting ZIP file: {str(e)}")
        
        if not extracted_files:
            raise Exception("No NetCDF files found in ZIP archive")
        
        return extracted_files
    
    def get_available_dates(self, start_date=None, end_date=None):
        """
        Get list of dates for which CAMS data is typically available
        Note: This doesn't check actual availability, just generates reasonable date range
        
        Parameters:
        start_date (str): Start date (default: 30 days ago)
        end_date (str): End date (default: yesterday)
        
        Returns:
        list: List of date strings in YYYY-MM-DD format
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        return [date.strftime('%Y-%m-%d') for date in date_range]
    
    def list_downloaded_files(self):
        """List all downloaded CAMS files"""
        downloaded_files = []

        for zip_file in self.download_dir.glob("*-cams.nc.zip"):
            date_str = zip_file.stem.replace("-cams.nc", "")
            # Check if extracted file exists
            extracted_file = self.extracted_dir / f"{date_str}-cams.nc"
            if extracted_file.exists():
                filename = extracted_file.name
            else:
                filename = f"{date_str}-cams.nc"  # Fallback filename

            file_info = {
                'date': date_str,
                'filename': filename,  # Add filename property
                'zip_path': str(zip_file),
                'size_mb': zip_file.stat().st_size / (1024 * 1024),
                'downloaded': zip_file.stat().st_mtime
            }
            downloaded_files.append(file_info)

        # Sort by date (newest first)
        downloaded_files.sort(key=lambda x: x['date'], reverse=True)
        return downloaded_files
    
    def cleanup_old_files(self, days_old=30):
        """
        Clean up downloaded files older than specified days
        
        Parameters:
        days_old (int): Delete files older than this many days
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            deleted_count = 0
            for zip_file in self.download_dir.glob("*-cams.nc.zip"):
                if datetime.fromtimestamp(zip_file.stat().st_mtime) < cutoff_date:
                    zip_file.unlink()
                    deleted_count += 1
            
            # Also clean extracted files
            for nc_file in self.extracted_dir.glob("*.nc"):
                if datetime.fromtimestamp(nc_file.stat().st_mtime) < cutoff_date:
                    nc_file.unlink()
                    deleted_count += 1
            
            print(f"🧹 Cleaned up {deleted_count} old files")
            return deleted_count
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            return 0


def test_cams_downloader():
    """Test function for CAMS downloader"""
    print("Testing CAMS downloader...")
    
    downloader = CAMSDownloader()
    
    if not downloader.is_client_ready():
        print("❌ CDS API client not ready. Please check your credentials.")
        return False
    
    # Test with recent date
    test_date = (datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d')
    
    print(f"Testing download for date: {test_date}")
    print("⚠️  This may take several minutes for the first download...")
    
    try:
        # Download data (will skip if already exists)
        zip_path = downloader.download_cams_data(test_date)
        print(f"✅ Download successful: {zip_path}")
        
        # Test extraction
        extracted_files = downloader.extract_cams_files(zip_path)
        print(f"✅ Extraction successful: {len(extracted_files)} files")
        
        # List downloaded files
        downloaded = downloader.list_downloaded_files()
        print(f"✅ Found {len(downloaded)} downloaded files")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_cams_downloader()