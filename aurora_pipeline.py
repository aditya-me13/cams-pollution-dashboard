# aurora_pipeline.py
# End-to-end pipeline for CAMS data → Aurora model → predictions → NetCDF
import subprocess
import os

def get_freest_cuda_device_id():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, encoding='utf-8'
        )
        memory_free = [int(x) for x in result.stdout.strip().split('\n')]
        device_id = memory_free.index(max(memory_free))
        return str(device_id)
    except Exception as e:
        print(f"Could not query nvidia-smi, defaulting to 0. Error: {e}")
        return "0"

# Set CUDA_VISIBLE_DEVICES before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = get_freest_cuda_device_id()


import torch
import xarray as xr
import pickle
from pathlib import Path
import numpy as np
import zipfile
import cdsapi
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from aurora import Batch, Metadata, AuroraAirPollution, rollout


class AuroraPipeline:
    def __init__(self, 
             extracted_dir="downloads/extracted", 
             static_path="static_vars.pkl", 
             model_ckpt="aurora-0.4-air-pollution.ckpt", 
             model_repo="microsoft/aurora",
             device=None):

        if device is None or device == "cuda":
            # CUDA_VISIBLE_DEVICES is set, so use 'cuda:0'
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.extracted_dir = Path(extracted_dir)
        self.static_path = Path(static_path)
        self.model_ckpt = model_ckpt
        self.model_repo = model_repo
        self.device = device
        self.static_vars = self._load_static_vars()
        self.model = None

    def _load_static_vars(self):
        """Load static variables from Hugging Face Hub"""
        static_path = hf_hub_download(
            repo_id="microsoft/aurora",
            filename="aurora-0.4-air-pollution-static.pickle",
        )
        if not Path(static_path).exists():
            raise FileNotFoundError(f"Static variables file not found: {static_path}")
        with open(static_path, "rb") as f:
            static_vars = pickle.load(f)
        return static_vars

    def create_batch(self, date_str, Batch, Metadata):
        """Create a batch for Aurora model from CAMS data"""
        surface_path = self.extracted_dir / f"{date_str}-cams-surface.nc"
        atmos_path = self.extracted_dir / f"{date_str}-cams-atmospheric.nc"
        if not surface_path.exists() or not atmos_path.exists():
            raise FileNotFoundError(f"Missing CAMS files for {date_str} in {self.extracted_dir}")

        surf_vars_ds = xr.open_dataset(surface_path, engine="netcdf4", decode_timedelta=True)
        atmos_vars_ds = xr.open_dataset(atmos_path, engine="netcdf4", decode_timedelta=True)

        # Select zero-hour forecast
        surf_vars_ds = surf_vars_ds.isel(forecast_period=0)
        atmos_vars_ds = atmos_vars_ds.isel(forecast_period=0)

        batch = Batch(
            surf_vars={
                "2t": torch.from_numpy(surf_vars_ds["t2m"].values[None]),
                "10u": torch.from_numpy(surf_vars_ds["u10"].values[None]),
                "10v": torch.from_numpy(surf_vars_ds["v10"].values[None]),
                "msl": torch.from_numpy(surf_vars_ds["msl"].values[None]),
                "pm1": torch.from_numpy(surf_vars_ds["pm1"].values[None]),
                "pm2p5": torch.from_numpy(surf_vars_ds["pm2p5"].values[None]),
                "pm10": torch.from_numpy(surf_vars_ds["pm10"].values[None]),
                "tcco": torch.from_numpy(surf_vars_ds["tcco"].values[None]),
                "tc_no": torch.from_numpy(surf_vars_ds["tc_no"].values[None]),
                "tcno2": torch.from_numpy(surf_vars_ds["tcno2"].values[None]),
                "gtco3": torch.from_numpy(surf_vars_ds["gtco3"].values[None]),
                "tcso2": torch.from_numpy(surf_vars_ds["tcso2"].values[None]),
            },
            static_vars={k: torch.from_numpy(v) for k, v in self.static_vars.items()},
            atmos_vars={
                "t": torch.from_numpy(atmos_vars_ds["t"].values[None]),
                "u": torch.from_numpy(atmos_vars_ds["u"].values[None]),
                "v": torch.from_numpy(atmos_vars_ds["v"].values[None]),
                "q": torch.from_numpy(atmos_vars_ds["q"].values[None]),
                "z": torch.from_numpy(atmos_vars_ds["z"].values[None]),
                "co": torch.from_numpy(atmos_vars_ds["co"].values[None]),
                "no": torch.from_numpy(atmos_vars_ds["no"].values[None]),
                "no2": torch.from_numpy(atmos_vars_ds["no2"].values[None]),
                "go3": torch.from_numpy(atmos_vars_ds["go3"].values[None]),
                "so2": torch.from_numpy(atmos_vars_ds["so2"].values[None]),
            },
            metadata=Metadata(
                lat=torch.from_numpy(atmos_vars_ds.latitude.values),
                lon=torch.from_numpy(atmos_vars_ds.longitude.values),
                time=(atmos_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[-1],),
                atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
            ),
        )
        return batch
    def load_model(self, AuroraAirPollution):
        """Load Aurora model and move to device"""
        import gc
        
        # Check memory BEFORE loading
        if torch.cuda.is_available():
            print(f"📊 GPU Memory BEFORE loading model:")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"   Reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            print(f"   Free:      {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3:.2f} GB")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        model = AuroraAirPollution()
        
        # Check AFTER initialization but BEFORE loading checkpoint
        if torch.cuda.is_available():
            print(f"📊 GPU Memory AFTER model init:")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        model.load_checkpoint(self.model_repo, self.model_ckpt)
        
        # Check AFTER loading checkpoint
        if torch.cuda.is_available():
            print(f"📊 GPU Memory AFTER checkpoint load:")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        model.eval()
        model = model.to(self.device)
        
        # Check AFTER moving to device
        if torch.cuda.is_available():
            print(f"📊 GPU Memory AFTER moving to device:")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"   Reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        self.model = model
        print(f"✅ Model loaded on {self.device}")
        return model

    # def predict(self, batch, rollout, steps=4):
    #     """Run model prediction for given batch and steps"""
    #     if self.model is None:
    #         raise RuntimeError("Model not loaded. Call load_model() first.")
    #     batch = batch  # Already on CPU; move to device if needed
    #     with torch.inference_mode():
    #         predictions = [pred.to("cpu") for pred in rollout(self.model, batch, steps=steps)]
    #     return predictions

    def predict(self, batch, rollout, steps=4):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Move batch to device
        batch = batch.to(self.device)  # ← Add this!
        
        with torch.inference_mode():
            predictions = [pred.to("cpu") for pred in rollout(self.model, batch, steps=steps)]
        print("predictions debug")
        print(type(predictions))
        print(dir(predictions[0]))                     # All attributes/methods

        # Access components
        # print(pred.surf_vars.keys())         # Dict keys
        # print(pred.surf_vars['pm2p5'].shape) # Shape of one variable
        # print(pred.atmos_vars['t'].shape)    # Shape of atmospheric var
        # print(pred.metadata.lat.shape)       # Latitude shape
        return predictions

    # def save_predictions_netcdf(self, predictions, output_path):
    #     """Save all surf_vars from predictions as NetCDF file"""
    #     output_path = Path(output_path)
    #     var_names = list(predictions[0].surf_vars.keys())
    #     data_vars = {}
    #     for var in var_names:
    #         arr = np.stack([pred.surf_vars[var].cpu().numpy() for pred in predictions], axis=0)
    #         data_vars[var] = (["step"] + [f"dim{i}" for i in range(1, arr.ndim)], arr)
    #     ds = xr.Dataset(data_vars)
    #     ds.to_netcdf(output_path)
    #     print(f"✅ Predictions saved to {output_path}")


    # def save_predictions_to_netcdf(self, predictions, output_path):
    #     """
    #     Save all prediction steps to a single NetCDF file with step dimension
        
    #     Parameters:
    #     predictions: List of Batch objects
    #     output_path: Path to save NetCDF file
    #     """
    #     import xarray as xr
    #     import numpy as np
    #     from pathlib import Path
        
    #     output_path = Path(output_path)
        
    #     print(f"\n{'='*60}")
    #     print(f"💾 Saving {len(predictions)} prediction steps to NetCDF")
    #     print(f"{'='*60}\n")
        
    #     # Get spatial shape
    #     spatial_shape = predictions[0].spatial_shape
    #     print(f"   Spatial shape: {spatial_shape}")
        
    #     # Extract metadata from first prediction
    #     metadata = predictions[0].metadata
    #     lats = metadata.lat.cpu().numpy() if hasattr(metadata.lat, 'cpu') else metadata.lat.numpy()
    #     lons = metadata.lon.cpu().numpy() if hasattr(metadata.lon, 'cpu') else metadata.lon.numpy()
        
    #     print(f"   Latitude points: {len(lats)}")
    #     print(f"   Longitude points: {len(lons)}")
        
    #     # Get pressure levels if available
    #     pressure_levels = None
    #     if hasattr(metadata, 'atmos_levels') and metadata.atmos_levels:
    #         pressure_levels = list(metadata.atmos_levels)
    #         print(f"   Pressure levels: {len(pressure_levels)}")
        
    #     # Collect all data
    #     all_datasets = []
        
    #     for step_idx, pred_batch in enumerate(predictions):
    #         print(f"   Processing step {step_idx + 1}/{len(predictions)}...")
            
    #         data_vars = {}
            
    #         # Extract surface variables
    #         if pred_batch.surf_vars:
    #             for var_name, var_tensor in pred_batch.surf_vars.items():
    #                 # Move to CPU and convert to numpy
    #                 var_data = var_tensor.cpu().numpy() if var_tensor.is_cuda else var_tensor.numpy()
                    
    #                 # Remove batch dimension: (1, lat, lon) -> (lat, lon)
    #                 if var_data.ndim == 3 and var_data.shape[0] == 1:
    #                     var_data = var_data[0]
                    
    #                 data_vars[var_name] = (["lat", "lon"], var_data)
            
    #         # Extract atmospheric variables
    #         if pred_batch.atmos_vars and pressure_levels:
    #             for var_name, var_tensor in pred_batch.atmos_vars.items():
    #                 # Move to CPU and convert to numpy
    #                 var_data = var_tensor.cpu().numpy() if var_tensor.is_cuda else var_tensor.numpy()
                    
    #                 # Remove batch dimension: (1, levels, lat, lon) -> (levels, lat, lon)
    #                 if var_data.ndim == 4 and var_data.shape[0] == 1:
    #                     var_data = var_data[0]
                    
    #                 data_vars[var_name] = (["pressure_level", "lat", "lon"], var_data)
            
    #         # Create coordinates for this step
    #         coords = {
    #             "lat": (["lat"], lats),
    #             "lon": (["lon"], lons),
    #         }
            
    #         if pressure_levels:
    #             coords["pressure_level"] = (["pressure_level"], pressure_levels)
            
    #         # Create dataset for this step
    #         ds = xr.Dataset(
    #             data_vars=data_vars,
    #             coords=coords,
    #             attrs={
    #                 "step": step_idx,
    #                 "description": f"Aurora air pollution prediction - Step {step_idx}",
    #             }
    #         )
            
    #         all_datasets.append(ds)
        
    #     # Combine all steps along 'step' dimension
    #     print(f"\n   Combining {len(all_datasets)} steps...")
    #     combined_ds = xr.concat(all_datasets, dim="step")
        
    #     # Add step coordinate
    #     combined_ds = combined_ds.assign_coords(step=np.arange(len(predictions)))
        
    #     # Add global attributes
    #     combined_ds.attrs.update({
    #         "title": "Aurora Air Pollution Predictions",
    #         "source": "Aurora model rollout",
    #         "model": "Aurora Air Pollution v0.4",
    #         "total_steps": len(predictions),
    #         "creation_date": str(np.datetime64('now')),
    #         "spatial_resolution": f"{len(lats)}x{len(lons)}",
    #     })
        
    #     # Save to NetCDF
    #     print(f"\n   Writing to: {output_path}")
    #     combined_ds.to_netcdf(output_path, mode='w', format='NETCDF4')
        
    #     # Print summary
    #     print(f"\n   File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    #     print(f"\n✅ Predictions saved successfully!")
    #     print(f"{'='*60}\n")

    # def save_predictions_to_netcdf(self, predictions, output_path):
    #     """Save all prediction steps to a single NetCDF file with step dimension"""
    #     for pred in predictions:
    #         pred.to_netcdf(output_path)
    def save_predictions_to_netcdf(self, predictions, output_path):
        """Save all prediction steps to a single NetCDF file compatible with visualization pipeline"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving {len(predictions)} prediction steps to {output_path}")

        try:
            # Try the new single-file method
            return self._save_predictions_single_file(predictions, output_path)
        except Exception as e:
            print(f"⚠️  Single file method failed: {e}")
            print(f"🔄 Falling back to original method...")
            return self._save_predictions_original_method(predictions, output_path)

    def _save_predictions_single_file(self, predictions, output_path):
        """Save all prediction steps to a single NetCDF file (new method)"""
        # Get metadata from first prediction
        first_pred = predictions[0]
        metadata = first_pred.metadata

        # Extract coordinates
        lats = metadata.lat.cpu().numpy() if hasattr(metadata.lat, 'cpu') else metadata.lat.numpy()
        lons = metadata.lon.cpu().numpy() if hasattr(metadata.lon, 'cpu') else metadata.lon.numpy()

        # Create step coordinate
        steps = np.arange(len(predictions))

        # Prepare data variables
        data_vars = {}
        coords = {
            'step': ('step', steps),
            'lat': ('lat', lats),
            'lon': ('lon', lons)
        }

        # Add surface variables
        surf_var_names = list(first_pred.surf_vars.keys())
        for var in surf_var_names:
            # Stack predictions along step dimension
            var_data_list = []
            for pred in predictions:
                var_tensor = pred.surf_vars[var]
                # Move to CPU and convert to numpy
                var_data = var_tensor.cpu().numpy() if hasattr(var_tensor, 'cpu') else var_tensor.numpy()

                # Robust dimension handling: squeeze all singleton dimensions and keep only last 2 (lat, lon)
                var_data = np.squeeze(var_data)  # Remove all singleton dimensions

                # Ensure we have exactly 2 dimensions (lat, lon) for surface variables
                if var_data.ndim > 2:
                    # Take the last 2 dimensions as lat, lon
                    var_data = var_data[..., :, :]
                    # If still more than 2D, take the first slice of extra dimensions
                    while var_data.ndim > 2:
                        var_data = var_data[0]
                elif var_data.ndim < 2:
                    raise ValueError(f"Surface variable {var} has insufficient dimensions: {var_data.shape}")

                var_data_list.append(var_data)

            # Stack along step dimension: (steps, lat, lon)
            arr = np.stack(var_data_list, axis=0)
            data_vars[var] = (['step', 'lat', 'lon'], arr)

        # Add atmospheric variables if present
        if hasattr(first_pred, 'atmos_vars') and first_pred.atmos_vars:
            atmos_levels = list(metadata.atmos_levels) if hasattr(metadata, 'atmos_levels') else None
            if atmos_levels:
                coords['pressure_level'] = ('pressure_level', atmos_levels)

                atmos_var_names = list(first_pred.atmos_vars.keys())
                for var in atmos_var_names:
                    var_data_list = []
                    for pred in predictions:
                        var_tensor = pred.atmos_vars[var]
                        # Move to CPU and convert to numpy
                        var_data = var_tensor.cpu().numpy() if hasattr(var_tensor, 'cpu') else var_tensor.numpy()

                        # Robust dimension handling: squeeze singleton dimensions but keep 3D structure
                        var_data = np.squeeze(var_data)  # Remove singleton dimensions

                        # Ensure we have exactly 3 dimensions (levels, lat, lon) for atmospheric variables
                        if var_data.ndim > 3:
                            # Take the last 3 dimensions as levels, lat, lon
                            var_data = var_data[..., :, :, :]
                            # If still more than 3D, take the first slice of extra dimensions
                            while var_data.ndim > 3:
                                var_data = var_data[0]
                        elif var_data.ndim < 3:
                            raise ValueError(f"Atmospheric variable {var} has insufficient dimensions: {var_data.shape}")

                        var_data_list.append(var_data)

                    # Stack along step dimension: (steps, levels, lat, lon)
                    arr = np.stack(var_data_list, axis=0)
                    data_vars[f"{var}_atmos"] = (['step', 'pressure_level', 'lat', 'lon'], arr)

        # Create dataset
        ds = xr.Dataset(data_vars, coords=coords)

        # Add global attributes
        ds.attrs.update({
            'title': 'Aurora Air Pollution Model Predictions',
            'source': 'Aurora model by Microsoft Research',
            'creation_date': datetime.now().isoformat(),
            'forecast_steps': len(predictions),
            'spatial_resolution': f"{abs(lons[1] - lons[0]):.3f} degrees",
            'conventions': 'CF-1.8'
        })

        # Add variable attributes for better visualization
        var_attrs = {
            '2t': {'long_name': '2 metre temperature', 'units': 'K'},
            '10u': {'long_name': '10 metre U wind component', 'units': 'm s-1'},
            '10v': {'long_name': '10 metre V wind component', 'units': 'm s-1'},
            'msl': {'long_name': 'Mean sea level pressure', 'units': 'Pa'},
            'pm1': {'long_name': 'Particulate matter d < 1 um', 'units': 'kg m-3'},
            'pm2p5': {'long_name': 'Particulate matter d < 2.5 um', 'units': 'kg m-3'},
            'pm10': {'long_name': 'Particulate matter d < 10 um', 'units': 'kg m-3'},
            'tcco': {'long_name': 'Total column carbon monoxide', 'units': 'kg m-2'},
            'tc_no': {'long_name': 'Total column nitrogen monoxide', 'units': 'kg m-2'},
            'tcno2': {'long_name': 'Total column nitrogen dioxide', 'units': 'kg m-2'},
            'gtco3': {'long_name': 'Total column ozone', 'units': 'kg m-2'},
            'tcso2': {'long_name': 'Total column sulphur dioxide', 'units': 'kg m-2'}
        }

        for var_name, attrs in var_attrs.items():
            if var_name in ds.data_vars:
                ds[var_name].attrs.update(attrs)

        # Save to NetCDF
        ds.to_netcdf(output_path, format='NETCDF4')
        print(f"✅ Predictions saved to {output_path}")
        print(f"   Variables: {list(ds.data_vars.keys())}")
        print(f"   Steps: {len(steps)}")
        print(f"   Spatial grid: {len(lats)}x{len(lons)}")

        return output_path

    def _save_predictions_original_method(self, predictions, output_path):
        """Fallback: Save predictions using the original method (separate files per step)"""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)

        for step, pred in enumerate(predictions):
            # Create xarray dataset for surface variables
            surf_data = {}
            for var_name, var_data in pred.surf_vars.items():
                surf_data[var_name] = (
                    ["time", "batch", "lat", "lon"],
                    var_data.cpu().numpy() if hasattr(var_data, 'cpu') else var_data.numpy()
                )

            # Create xarray dataset for atmospheric variables
            atmos_data = {}
            for var_name, var_data in pred.atmos_vars.items():
                atmos_data[var_name] = (
                    ["time", "batch", "level", "lat", "lon"],
                    var_data.cpu().numpy() if hasattr(var_data, 'cpu') else var_data.numpy()
                )

            # Create surface dataset
            surf_ds = xr.Dataset(
                surf_data,
                coords={
                    "time": [pred.metadata.time[0]],
                    "batch": [0],
                    "lat": pred.metadata.lat.cpu().numpy() if hasattr(pred.metadata.lat, 'cpu') else pred.metadata.lat.numpy(),
                    "lon": pred.metadata.lon.cpu().numpy() if hasattr(pred.metadata.lon, 'cpu') else pred.metadata.lon.numpy(),
                }
            )

            # Create atmospheric dataset
            atmos_ds = xr.Dataset(
                atmos_data,
                coords={
                    "time": [pred.metadata.time[0]],
                    "batch": [0],
                    "level": list(pred.metadata.atmos_levels),
                    "lat": pred.metadata.lat.cpu().numpy() if hasattr(pred.metadata.lat, 'cpu') else pred.metadata.lat.numpy(),
                    "lon": pred.metadata.lon.cpu().numpy() if hasattr(pred.metadata.lon, 'cpu') else pred.metadata.lon.numpy(),
                }
            )

            # Save to NetCDF
            surf_filename = f"step_{step:02d}_surface.nc"
            atmos_filename = f"step_{step:02d}_atmospheric.nc"

            surf_ds.to_netcdf(output_dir / surf_filename)
            atmos_ds.to_netcdf(output_dir / atmos_filename)

            print(f"Saved step {step} predictions (fallback method)")

        return output_dir

    def run_pipeline(self, date_str, Batch, Metadata, AuroraAirPollution, rollout, steps=4, output_path=None):
        """Full pipeline: batch creation, model loading, prediction, save output"""
        batch = self.create_batch(date_str, Batch, Metadata)
        self.load_model(AuroraAirPollution)
        predictions = self.predict(batch, rollout, steps=steps)
        if output_path:
            self.save_predictions_to_netcdf(predictions, output_path)
        return predictions

# Example usage (not run on import)
if __name__ == "__main__":
    # from aurora_model import AuroraAirPollution, Batch, Metadata, rollout
    # pipeline = AuroraPipeline()
    # predictions = pipeline.run_pipeline(
    #     date_str="2023-10-01",
    #     Batch=Batch,
    #     Metadata=Metadata,
    #     AuroraAirPollution=AuroraAirPollution,
    #     rollout=rollout,
    #     steps=4,
    #     output_path="predictions_2023-10-01.nc"
    # )
    pass