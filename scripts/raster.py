# -*- coding: utf-8 -*-
"""
__author__ = "Augustine M Gbondo"
__credits__ = ["Augustine M Gbondo"]
__project__ = Geogenic radon potential mapping in Hessen using machine learning techniques
__maintainer__ = "Augustine M Gbondo"
__email__ = "gbondomadaar@gmail.com"
"""

import requests
import pandas as pd
import zipfile
import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from pyproj import CRS
from rasterio.fill import fillnodata
import scipy.ndimage
import rioxarray as rx
from rasterio.io import MemoryFile
from matplotlib.colors import Normalize
import math
import shutil
from datetime import datetime
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from datetime import datetime
from rasterio.features import shapes
import rasterio as rio
import rasterio.mask
import fiona
from rasterio import open as rio_open
from rasterio.features import geometry_mask
from shapely.geometry import mapping
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize



class DirectoryManager:
    def __init__(self):
        pass
    
    def list_dir_folders(self, path):
        if os.path.exists(path):
            for f in os.scandir(path):
                print(f.name)
        else:
            print(f"The directory {path} does not exist.")

    def create_directory(self, path):
        """Create a directory if it does not exist yet"""
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Directory created or already exists: {path}")
        except OSError as err:
            print(f"Warning: {err}")
        return path

    def create_subfolders(self, base_path, subfolders):
        """
        Creates specified subfolders under the base path.

        Parameters:
        base_path (str): The path to the base directory where subfolders
        will be created.
        subfolders (list): List of subfolder names to create.

        Returns:
        dict: Dictionary containing paths to the created subfolders.
        """
        # Dictionary to hold the full paths to the subfolders
        paths = {}

        for subfolder in subfolders:
            # Construct the full path for each subfolder
            subfolder_path = os.path.join(base_path, subfolder)
            # Create the subfolder if it does not exist
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
            # Add the subfolder path to the dictionary
            paths[subfolder] = subfolder_path

        return paths

        #Location_maps=subfolder_paths['Location map']
        #OpenStreet_maps=subfolder_paths['OpenStreet map']
        #AerielImagery_maps=subfolder_paths['AerielImage map']
        #Factsheet_PDF=subfolder_paths['Factsheet_PDF']
        #Factsheet_docx=subfolder_paths['Factsheet_docx']

    def download_and_extract_zip(self, url, target_dir, filename:str):
        """
        Download a zip file from the given URL and extract it to the target directory.
        """
        filename=filename
        response = requests.get(url)
        zip_file_path = os.path.join(target_dir, filename)

        with open(zip_file_path, 'wb') as f:
            f.write(response.content)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

        os.remove(zip_file_path)


    # Download and extract the zip file
    #download_and_extract_zip(url, target_dir, 'srtm_germany_dtm.zip')

    def remove_directory(self, path):
        """Remove directory"""
        try:
            shutil.rmtree(path, ignore_errors=True)
            print(f"Directory removed: {path}")
        except OSError as err:
            print(f"Warning: {err}")

    def validate_file(self, file):
        """Assert file exists and is a file"""
        assert os.path.exists(file) and os.path.isfile(
            file), f"File does not exist or is not a file: {file}"

    def print_start_time(self, func):
        """Print start time of process"""
        self._print_wrapper()
        self.start = datetime.now()
        print(f"Start: {func}: {self.start}")
        self._print_wrapper()
        return self.start

    def print_end_time(self, func):
        """Print run time of process"""
        self._print_wrapper()
        end = datetime.now() - self.start
        print(f"End: {func}: {end}")
        self._print_wrapper()

    def _print_wrapper(self):
        """Print dash line"""
        print(f"{'-' * 80}")

# modified from https://github.com/OmdenaAI/omdena-philippines-renewable/blob/main/src/tasks/task-2-nightlight-processing/helpers/raster.py   
# .................................................................

class RasterProperties:
    def __init__(self, file_path):
        """Initialize the RasterProcessor with a file path."""
        self.file_path = file_path
        self.raster = self.open_raster()

    def open_raster_rio(self, masked=True):
        """Open raster file using rioxarray."""
        self.validate_file(self.file_path)  # Validate the file before opening
        return rx.open_rasterio(self.file_path, masked=masked)

    def open_raster(self):
        """Open raster using rasterio."""
        self.validate_file(self.file_path)  # Validate the file before opening
        return rasterio.open(self.file_path)

    def validate_file(self, file_path):
        """Validate if the file exists."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
    
    @staticmethod
    def check_tifs_CRS(folder_path):
        """Check the CRS of all TIFF files in the folder."""
        tiff_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))]

        if not tiff_files:
            print("No TIFF files found in the folder.")
            return

        for tiff_file in tiff_files:
            tiff_path = os.path.join(folder_path, tiff_file)
            try:
                with rasterio.open(tiff_path) as src:
                    crs = src.crs
                    print(f"{tiff_file}: CRS = {crs}")
            except Exception as e:
                print(f"Error reading {tiff_file}: {e}")

    def raster_statistics(self):
        """Return raster's min, max, and average pixel values."""
        try:
            data = self.raster.read(1)  
            min_val = data.min()
            max_val = data.max()
            mean_val = data.mean()
        except AttributeError as err:
            print(f"Error: {err}")
            return None, None, None
        return min_val, max_val, mean_val

    def raster_bounds(self):
        """Return raster's bounding box."""
        return self.raster.bounds

    def raster_dimensions(self):
        """Return raster's width and height."""
        return self.raster.width, self.raster.height

    def raster_crs(self):
        """Coordinate system of the raster file."""
        return self.raster.crs

    def raster_resolution(self):
        """Pixel resolution of the raster file."""
        return self.raster.res

    def rasterio_dataset_to_file(self, data, file, profile, dtype="float32", nodata=0, band=1):
        """Create a raster file from rasterio's dataset."""
        with rasterio.Env():
            profile.update(
                dtype=dtype,
                count=band,
                compress='lzw',
                nodata=nodata
            )
            with rasterio.open(file, 'w', **profile) as dst:
                dst.write(data.astype(rasterio.float32), 1)
        return file

    def get_dataset_profile(self):
        """Get rasterio's dataset profile."""
        return self.raster.profile

    def read_raster_band(self, band=1):
        """Read specific band of raster."""
        return self.raster.read(band)

    def get_raster_transform_properties(self):
        """Get raster properties to use for transformation."""
        return self.raster.transform

    def get_raster_properties_rio(self):
        """Wrapper function to get all pertinent properties of raster using rioxarray."""
        return {
            "stats": self.raster_statistics(),
            "bounds": self.raster_bounds(),
            "dimensions": self.raster_dimensions(),
            "crs": self.raster_crs(),
            "pixel_res": self.raster_resolution()
        }


def check_tifs_CRS(folder_path):
        """Check the CRS of all TIFF files in the folder."""
        tiff_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))]

        if not tiff_files:
            print("No TIFF files found in the folder.")
            return

        for tiff_file in tiff_files:
            tiff_path = os.path.join(folder_path, tiff_file)
            try:
                with rasterio.open(tiff_path) as src:
                    crs = src.crs
                    print(f"{tiff_file}: CRS = {crs}")
            except Exception as e:
                print(f"Error reading {tiff_file}: {e}")

    
#------------------------------------------------------------------------------#

class RasterProcessor:
    def __init__(self, file_path=None, output_dir=None):
        """
        Initialize the RasterProcessor with a file path and output directory.
        """
        self.file_path = file_path
        self.output_dir = output_dir

        if file_path:
            self.raster = self.open_raster()

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def open_raster(self):
        """
        Open raster using rasterio.
        """
        return rio_open(self.file_path)

    def define_projection(self, raster_file, crs, output_file):

        """
        Define or update the projection of a raster file and save the result.
        """
        with rasterio.open(raster_file) as src:
            meta = src.meta.copy()
            meta.update({
                'crs': crs
            })
            with rasterio.open(output_file, 'w', **meta) as dst:
                dst.write(src.read())
                dst.write_colormap(1, src.colormap(1))

    def reproject_raster(self, raster_file, output_file):

        crs='EPSG:25832'
        """
        Reproject a raster file to a new CRS and save the result.
        """
        with rasterio.open(raster_file) as src:
            transform, width, height = calculate_default_transform(
                src.crs, crs, src.width, src.height, *src.bounds)
            meta = src.meta.copy()
            meta.update({
                'crs': crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(output_file, 'w', **meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=Resampling.nearest)

    def resample_raster(self, input_raster, output_raster):

        target_resolution = (250, 250)
        """
        Resample a raster to the target resolution.
        """
        with rasterio.open(input_raster) as src:
            # Read the metadata of the input raster
            metadata = src.meta.copy()

            # Calculate new dimensions based on target resolution
            # Target resolution in meters
            target_width, target_height = target_resolution

            # Compute new dimensions of the raster in pixels
            scale_x = src.res[0] / target_width
            scale_y = src.res[1] / target_height
            new_width = int(src.width * scale_x)
            new_height = int(src.height * scale_y)

            # Update metadata for the new dimensions and resolution
            metadata.update({
                'width': new_width,
                'height': new_height,
                'transform': src.transform * src.transform.scale(
                    (src.width / new_width),
                    (src.height / new_height)
                )
            })

            # Perform the resampling
            with rasterio.open(output_raster, 'w', **metadata) as dst:
                for band in src.indexes:
                    data = src.read(band)
                    resampled_data = np.empty((new_height, new_width), dtype=data.dtype)

                    # Resample the data
                    resampled_data = src.read(
                        band,
                        out_shape=(new_height, new_width),
                        resampling=Resampling.bilinear
                    )

                    # Write the resampled data to the output raster
                    dst.write(resampled_data, band)
                    
    
    def clip_raster_gpkg(self, raster_file, vector_file_path, output_file):
        """
        Clip a raster using a shapefile and save the result with no-data value set to zero.
        """
        # Read the shapefile
        #shapes = [mapping(geom) for geom in geodf.geometry]
        
        with fiona.open(vector_file_path, layer='Hessen', driver='GPKG') as vector_file:
                shapes = [feature["geometry"] for feature in vector_file]
                
        # Clip the raster with the shapefile
        with rasterio.open(raster_file) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
            out_meta = src.meta.copy()

        # Update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": 0  # Set no-data value to zero
        })

        # Write the clipped raster to file with no-data value set to zero
        with rasterio.open(output_file, "w", **out_meta) as dest:
            # Set the no-data value to zero
            out_image = np.where(np.isnan(out_image), 0, out_image)
            dest.write(out_image.astype(rasterio.float32))

            
    def clip_raster(self, raster_file, shp_file_path, output_file):
        """
        Clip a raster using a shapefile and save the result with no-data value set to zero.
        """
        # Read the shapefile
        with fiona.open(shp_file_path, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]

        # Clip the raster with the shapefile
        with rasterio.open(raster_file) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
            out_meta = src.meta.copy()

        # Update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": 0  # Set no-data value to zero
        })

        # Write the clipped raster to file with no-data value set to zero
        with rasterio.open(output_file, "w", **out_meta) as dest:
            # Set the no-data value to zero
            out_image = np.where(np.isnan(out_image), 0, out_image)
            dest.write(out_image.astype(rasterio.float32))

    def mask_raster(self, input_raster_path, boundary, output_raster_path):

        # Ensure boundary is in the same coordinate reference system (CRS) as the raster
        with rasterio.open(input_raster_path) as src:
            boundary = boundary.to_crs(src.crs)
            
            # Extract the geometry of the boundary to use as a mask
            boundary_geometry = [feature['geometry'] for feature in boundary.__geo_interface__['features']]
            
            # Clip the raster using the mask
            out_image, out_transform = rasterio.mask.mask(src, boundary_geometry, crop=True)
            
            # Update metadata for the clipped raster
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": src.nodata if src.nodata is not None else -9999
            })

        # Write the clipped raster to a new file
        with rasterio.open(output_raster_path, 'w', **out_meta) as dest:
            dest.write(out_image)

    def fill_raster_nodata(self, input_path, output_path ):
        """
        Fill NoData values in a raster using rasterio's 
        fillnodata function and save the result.
        """
        with rasterio.open(input_path) as src:
            profile = src.profile  # Copy the input file's profile (metadata)
            arr = src.read(1)  # Read the first band of the raster
            mask = src.read_masks(1)  # Read the mask (valid data pixels)

            # Fill the NoData values in the array
            arr_filled = fillnodata(
                arr, mask=mask, max_search_distance=10, smoothing_iterations=0)

            # Update the profile to remove NoData value as the gaps will be filled
            profile.update({
                'nodata': None
            })

            # Write the filled array to the output file
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(arr_filled, 1)


    def mask_raster_lite(self, band=1, op="lte", val=0):
        """Create a masked raster."""
        data = self.read_raster_band(band=band)
        mask = self.get_mask_operator(data, val, op)
        return np.ma.masked_array(data, mask=mask)

    def get_mask_operator(self, data, val, op):
        """Get mask operator."""
        operators = {
            "lte": data <= val,
            "lt": data < val,
            "ne": data != val,
            "eq": data == val,
            "gte": data >= val,
            "gt": data > val,
        }
        return operators[op]

    def fill_raster_mask(self, val=0):
        """Fill raster mask."""
        return np.ma.filled(self.raster.astype(float), val)

    def add_two_rasters(self, raster2):
        """Add two rasters."""
        try:
            return self.raster + raster2
        except Exception as err:
            print(f"err: {err}")
            return None

    def multiply_two_rasters(self, raster2):
        """Multiply two rasters."""
        try:
            return self.raster * raster2
        except Exception as err:
            print(f"err: {err}")
            return None

    def clip_raster_to_geom(self, geom, crs=None):
        """Clip raster within geometry bounds using rioxarray."""
        crs = crs if crs else "EPSG:4326"
        return self.raster.rio.clip(geometries=geom, crs=crs, drop=True, invert=False)

    def reproject_match_raster(self, match):
        """Reproject input raster using a match raster."""
        return self.raster.rio.reproject_match(match)

    def save_to_file(self, path, filename):
        """Save raster to file using rioxarray."""
        file = os.path.join(path, filename)
        self.rioxarray_to_file(file)
        return file

    def create_raster_shapes(self, properties, mask=None):
        """Create geojson shapes from raster."""
        with rasterio.Env():
            data = ({
                'properties': {'raster_val': v},
                'geometry': s
            } for i, (s, v) in enumerate(shapes(self.raster, mask=mask, transform=properties)))
        return data

    def rioxarray_to_file(self, file):
        """Export rioxarray data to file."""
        self.raster.rio.to_raster(file, compress='lzw', tiled=True)
        return file
    
#-----------------------------------------------------------------------------------------------#

class RasterPlotter:
        """
        Plot single or batch rasters using various methods.
        """
    def __init__(self, colormap='terrain', figsize=(10, 8), columns=3):
        self.colormap = colormap
        self.figsize = figsize
        self.columns = columns

    def raster_crs(self, raster_file):
        with rasterio.open(raster_file) as src:
            crs = src.crs
            print(f"{os.path.basename(raster_file)}: CRS = {crs}")

    def print_raster_metadata(self, raster_file):
        with rasterio.open(raster_file) as src:
            # Retrieve metadata
            metadata = {
                'Driver': src.driver,
                'Count': src.count,
                'CRS': src.crs,
                'Transform': src.transform,
                'Width': src.width,
                'Height': src.height,
                'Bounds': src.bounds,
                'NoData': src.nodata,
                'Data Type': src.dtypes[0]
            }

            # Extract resolution
            transform = src.transform
            resolution_x = transform[0]
            resolution_y = -transform[4]  # Inverted y resolution

            print(f"Metadata for raster file '{os.path.basename(raster_file)}':")
            for key, value in metadata.items():
                print(f"{key}: {value}")

            print(f"Resolution (pixel size):")
            print(f"X resolution: {resolution_x} meters/pixel")
            print(f"Y resolution: {resolution_y} meters/pixel")

    def plot_rasters(self, folder_path):
        """
        Plot all raster images from a folder in a grid layout with legends and tight layout.
        """
        raster_files = [f for f in os.listdir(
            folder_path) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
        num_rasters = len(raster_files)

        if num_rasters == 0:
            print("No raster files found in the folder.")
            return

        rows = int(np.ceil(num_rasters / self.columns))

        # Create a figure with a specific size
        plt.figure(figsize=self.figsize)

        for i, raster_file in enumerate(raster_files):
            raster_path = os.path.join(folder_path, raster_file)
            raster_name= os.path.splitext(os.path.basename(raster_file))[0]
            plt.subplot(rows, self.columns, i + 1)
            try:
                self.plot_raster(raster_path)
            except Exception as e:
                print(f"Error plotting {raster_file}: {e}")

        # Adjust layout
        plt.subplots_adjust(wspace=0, hspace=0)  
        plt.tight_layout(pad=0.1) 
        plt.title(raster_name)
        plt.axis('off')  

        plt.show()

        
    def plot_downsample_rasters(self, folder_path):
        """
        Downsample and plot all raster in a folder 
        """
        raster_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))]
        num_rasters = len(raster_files)

        if num_rasters == 0:
            print("No raster files found in the folder.")
            return

        rows = int(np.ceil(num_rasters / self.columns))

                # Create a figure with a specific size
        plt.figure(figsize=self.figsize)

        for i, raster_file in enumerate(raster_files):
            raster_path = os.path.join(folder_path, raster_file)
            raster_name= os.path.splitext(os.path.basename(raster_file))[0]
            plt.subplot(rows, self.columns, i + 1)
            try:
                self.plot_downsample_raster(raster_path, downsample_factor=10)
            except Exception as e:
                print(f"Error plotting {raster_file}: {e}")
                
            
        plt.subplots_adjust(wspace=0, hspace=0)  # Adjust gaps between subplots
        plt.tight_layout(pad=0.1)  # Adjust padding as needed
        plt.title(raster_name)
        plt.show()

    def plot_downsample_raster(self, raster_file, downsample_factor=10):
        """
        Plot a single downsampled raster image.
        """
        try:
            with rasterio.open(raster_file) as src:
                # Downsample the raster using read with windowed reading
                array = src.read(1, out_shape=(
                    src.count,
                    int(src.height // downsample_factor),
                    int(src.width // downsample_factor)
                ), resampling=rasterio.enums.Resampling.nearest)

                array = np.ma.masked_equal(array, src.nodata)  # Mask NoData values
                raster_name = os.path.splitext(os.path.basename(raster_file))[0]

                vmin, vmax = np.nanmin(array), np.nanmax(array)

                # Get the coordinate system
                transform = src.transform * src.transform.scale(
                    (src.width / array.shape[-1]),
                    (src.height / array.shape[-2])
                )

                extent = (transform[2], transform[2] + transform[0] * src.width,
                          transform[5] + transform[4] * src.height, transform[5])
                
                #plt.figure(figsize=(10, 8))
                #plt.figure(figsize=self.figsize)

                plt.imshow(array, cmap=self.colormap,
                           norm=Normalize(vmin=vmin, vmax=vmax), extent=extent)
                plt.colorbar(label='values', orientation='vertical',
                             shrink=0.2, ticks=[vmin, vmax])  # Add colorbar with label
                plt.title(raster_name)
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.axis('off')  # Turn off axis labels
                # Set transparent background (optional)
                #plt.gca().patch.set_alpha(0)

                #plt.show()

        except Exception as e:
            print(f"Error plotting raster file {raster_file}: {e}")
    
    #@staticmethod
    def plot_raster_mask(self, raster_file):
        """
        Mask raster and plot.
        """
        try:
            with rasterio.open(raster_file) as src:
                # Read the first band and mask nodata values
                array = src.read(1, masked=True)
                nodata_value = 0
                array = np.ma.masked_equal(array, nodata_value)
                #array = np.ma.masked_equal(array, src.nodata)  # Mask NoData values
                raster_name = os.path.splitext(os.path.basename(raster_file))[0]

                vmin, vmax = np.nanmin(array), np.nanmax(array)
                transform = src.transform
                extent = (transform[2], transform[2] + transform[0] * src.width,
                          transform[5] + transform[4] * src.height, transform[5])

                plt.figure(figsize=(11, 8))
                plt.imshow(array, cmap=self.colormap, 
                           norm=Normalize(vmin=vmin, vmax=vmax), extent=extent)
                plt.colorbar(label='Values', orientation='vertical', 
                             shrink=0.3, ticks=[vmin, vmax])
                plt.title(raster_name)
                plt.axis('off')  
                plt.gca().patch.set_alpha(0)  
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                plt.show()

        except Exception as e:
            print(f"Error plotting raster file {raster_file}: {e}")


    def plot_raster(self, raster_file, cmap=None):
        """
        Plot a single raster image with a colormap.
        """
        try:
            with rasterio.open(raster_file) as src:
                array = src.read(1, masked=True)  # Read the first band and mask nodata values
                array = np.ma.masked_equal(array, src.nodata)  # Mask NoData values
                raster_name= os.path.splitext(os.path.basename(raster_file))[0]
                vmin, vmax = np.nanmin(array), np.nanmax(array)

                # Get the coordinate system
                transform = src.transform
                extent = (transform[2], transform[2] + transform[0] * src.width,
                          transform[5] + transform[4] * src.height, transform[5])
                
                plt.figure(figsize=(12, 9))
                plt.imshow(array, cmap=cmap if cmap is not None else self.colormap, 
                           norm=Normalize(vmin=vmin, vmax=vmax), extent=extent)
                plt.colorbar(label='values', orientation='vertical',
                             shrink=0.3, ticks=[vmin, vmax])  # Add colorbar with label
                plt.title(raster_name)
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.axis('off')  # Turn off axis labels
                # Set transparent background (optional)
                plt.gca().patch.set_alpha(0)

                plt.show()

        except Exception as e:
            print(f"Error plotting raster file {raster_file}: {e}")

            
    def plot_raster_bands(self, raster_file, band_names=None, band=None):
        """
        Plot all raster in a raster composite band. Option to ionclude band names. 
        """
        try:
            # Open the raster file
            with rasterio.open(raster_file) as src:
                if band is not None:
                    # If a specific band is selected, plot that band
                    array = src.read(band, masked=True)  # Read the specified band and mask nodata values
                    vmin, vmax = np.nanmin(array), np.nanmax(array)

                    # Extract the band name (description) or use a custom band name
                    if band_names and band <= len(band_names):
                        band_name = band_names[band - 1]
                    else:
                        band_name = src.descriptions[band - 1] if src.descriptions[band - 1] else f'Band {band}'

                    # Plot the specified band
                    plt.figure(figsize=(10, 8))
                    plt.imshow(array, cmap=self.colormap, norm=Normalize(vmin=vmin, vmax=vmax))
                    plt.colorbar(label=f'{band_name} values',
                    orientation='vertical',
                    shrink=0.8,
                    ticks=[vmin, vmax])
                    
                    plt.title(f'{band_name}')
                    plt.axis('off')
                    plt.show()

                else:
                    # Plot all bands as subplots if no specific band is provided
                    n_bands = src.count  # Number of bands in the raster
                    n_cols = self.columns  # Set the number of columns (up to 3 columns)
                    n_rows = int(np.ceil(n_bands / n_cols))  # Calculate the number of rows

                    fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize)
                    axes = axes.ravel()  # Flatten the axes array

                    for i in range(n_bands):
                        array = src.read(i + 1, masked=True)  # Read each band
                        vmin, vmax = np.nanmin(array), np.nanmax(array)

                        # Extract the band name (description) or use a custom band name
                        if band_names and i < len(band_names):
                            band_name = band_names[i]
                        else:
                            band_name = src.descriptions[i] if src.descriptions[i] else f'Band {i + 1}'

                        # Plot each band in a subplot
                        ax = axes[i]
                        img = ax.imshow(array, cmap=self.colormap, norm=Normalize(vmin=vmin, vmax=vmax))
                        ax.set_title(f'{band_name}')
                        ax.axis('off')

                        # Add a colorbar for each subplot
                        plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.3, ticks=[vmin, vmax])

                    # Hide any unused subplots (in case of unequal n_bands and grid size)
                    for j in range(i + 1, len(axes)):
                        axes[j].axis('off')

                    plt.tight_layout(pad=0.3)  # Adjust padding as needed
                    plt.show()

        except Exception as e:
            print(f"Error plotting raster file: {e}")

        
#--------------------------------------------------------------------------#            
class RasterCalculator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def open_raster(self, raster_file):
        """
        Open a raster file and return the source object and its metadata.
        """
        src = rasterio.open(raster_file)
        return src, src.meta

    def write_raster(self, array, meta, output_file):
        """
        Write an array to a raster file with the given metadata.
        """
        meta.update(dtype=rasterio.float32, count=1)
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(array.astype(rasterio.float32), 1)

    def average_rasters(self, raster_files, output_file):
        """
        Compute the average of multiple rasters and save the result.
        """
        arrays = []
        for raster_file in raster_files:
            src, _ = self.open_raster(raster_file)
            arrays.append(src.read(1, masked=True))

        # Compute the average
        avg_array = np.mean(np.array(arrays), axis=0)

        # Write the averaged raster to file
        _, meta = self.open_raster(raster_files[0])
        self.write_raster(avg_array, meta, output_file)

    def divide_raster(self, raster_file, divisor, output_file):
        """Divide a raster by a given divisor and save the result."""
        src, meta = self.open_raster(raster_file)
        divided_array = src.read(1, masked=True) / divisor

        # Write the divided raster to file
        self.write_raster(divided_array, meta, output_file)

    def add_rasters(self, raster_files, output_file):
        """Compute the sum of multiple rasters and save the result."""
        arrays = []
        for raster_file in raster_files:
            src, _ = self.open_raster(raster_file)
            arrays.append(src.read(1, masked=True))

        # Compute the sum
        sum_array = np.sum(np.array(arrays), axis=0)

        # Write the summed raster to file
        _, meta = self.open_raster(raster_files[0])
        self.write_raster(sum_array, meta, output_file)

    def weighted_overlay(self, raster_paths, weights, output_path):
      if len(raster_paths) != len(weights):
          raise ValueError("The number of raster paths must match the number of weights.")

      # Initialize variables for weighted sum
      weighted_sum = None
      profile = None

      for idx, raster_path in enumerate(raster_paths):
          with rio_open(raster_path) as src:
              # Read the raster data as a numpy array
              raster_data = src.read(1)
              # Initialize weighted sum array on the first iteration
              if weighted_sum is None:
                  weighted_sum = np.zeros_like(raster_data, dtype=np.float32)
                  profile = src.profile

              # Add the weighted raster to the weighted sum
              weighted_sum += raster_data * weights[idx]

      # Normalize the weighted sum if necessary (optional)
      # weighted_sum /= sum(weights)

      # Save the weighted sum to a new raster file
      profile.update(dtype=rasterio.float32, count=1)
      with rio_open(output_path, 'w', **profile) as dst:
          dst.write(weighted_sum, 1)

      print(f"Weighted overlay completed. Output saved to {output_path}")


    def classify_raster(self, raster_file, class_boundaries):

        src, _ = self.open_raster(raster_file)
        raster_data = src.read(1, masked=True)

        low_risk = (raster_data <= class_boundaries[0])
        medium_risk = (raster_data > class_boundaries[0]) & (raster_data <= class_boundaries[1])
        high_risk = (raster_data > class_boundaries[1])

        # Create an empty array for the classified data
        classified_raster = np.zeros_like(raster_data, dtype=np.uint8)

        # Assign class values
        classified_raster[low_risk] = 1
        classified_raster[medium_risk] = 2
        classified_raster[high_risk] = 3

        # Step 3: Plot the raster with a legend
        # Define colors for the classes
        cmap = mcolors.ListedColormap(['green', 'orange', 'red'])
        norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)

        class_labels = [
            f'Low risk: < {class_boundaries[0]}',
            f'Medium risk: {class_boundaries[0]} - {class_boundaries[1]}',
            f'High risk: > {class_boundaries[1]}'
        ]

        plt.figure(figsize=(10, 8))
        plt.imshow(classified_raster, cmap=cmap, norm=norm)
        #plt.colorbar(ticks=[1, 2, 3], label='Classes')


        colors = ['green', 'orange', 'red']

        # Create a bar plot
        #plt.figure(figsize=(8, 6))
        #plt.bar(class_labels, aggregated_values, color=colors)

        # Customize the legend with square symbols using zip
        legend_elements = [Patch(facecolor=color, edgecolor='black', label=label)
                          for label, color in zip(class_labels, colors)]

        plt.legend(handles=legend_elements, loc='lower right', fontsize=8.5)
        plt.title('Classified Raster Data')
        plt.axis('off')  # Turn off axis labels
        plt.show()
