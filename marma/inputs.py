from __future__ import annotations

import numpy as np
import rasterio as rio
import xdem
from pathlib import Path
import geoutils as gu
import pandas as pd
import pathlib


def parse_txt_dem(filepath: str | Path, crs: int | str | rio.crs.CRS = 3006, resolution_rounding: int = 1) -> xdem.DEM:
    """
    Parse an XYZ DEM text file (rows of "X Y Z" coordinates).

    The coordinates are assumed to represent the center of the pixel.

    For some reason, the resolution has to be rounded to obtain the right number. I'm not sure why...

    :param filepath: The path to the DEM text file.
    :param crs: The coordinate reference system to assign.
    :param resolution_rounding: The number of decimals to round the resolution.

    :returns: The parsed DEM object.
    """
    points = np.loadtxt(filepath)

    height = np.unique(points[:, 1]).shape[0]
    width = np.unique(points[:, 0]).shape[0]

    resolution = round((points[:, 0].max() - points[:, 0].min()) / width, resolution_rounding)

    # Obtain the indices for where to assign the coordinates in the output image.
    row_n = (height - 1) - ((points[:, 1] - points[:, 1].min()) / resolution).astype(int)
    col_n = ((points[:, 0] - points[:, 0].min()) / resolution).astype(int)
    indices = row_n * width + col_n

    # Initialize the image.
    data = np.zeros((1, height, width), dtype="float32") - 9999

    # Assign the Z values in the correct spots.
    data.ravel()[indices] = points[:, 2]

    transform = rio.transform.from_origin(
        west=points[:, 0].min() - resolution / 2,
        north=points[:, 1].max() + resolution / 2,
        xsize=resolution,
        ysize=resolution,
    )
    dem = xdem.DEM.from_array(data, transform, crs=crs, nodata=-9999)

    return dem


def load_dem(filepath: str | Path, conform_to: None | xdem.DEM) -> xdem.DEM:
    
    dem = xdem.DEM(filepath, load_data=False)
    if conform_to is not None:
        dem.crop(conform_to)
        dem.load()
        dem = dem.reproject(conform_to, resampling="bilinear")
    else:
        dem.load()
    return dem

def load_snow_and_ice(glacier_path: str | Path, snow_path: str | Path) -> gu.Vector:
    glaciers = gu.Vector(glacier_path)
    snow = gu.Vector(snow_path)

    glaciers.ds["type"] = "glacier"
    snow.ds["type"] = "snow"

    merged = gu.Vector(pd.concat([glaciers.ds, snow.ds]))
    merged.crs = glaciers.crs

    return merged


def load_all_inputs(gis_data_path: Path, temp_path: Path, reference_grid_year: int) -> tuple[dict[int, xdem.DEM], gu.Vector]:
    
    dems: dict[int, xdem.DEM] = {}

    text_years = [1959, 1978, 1991, 2008]
    available_years = text_years + [2016, 2021]
    def load_one_dem(year: int) -> xdem.DEM:
        if year in text_years:
            dem = parse_txt_dem(gis_data_path.joinpath(f"rasters/marm{year}10m.txt"))
            if year != reference_grid_year:
                dem = dem.reproject(dems[reference_grid_year])
            return dem
        if year == 2016:
            return load_dem(gis_data_path.joinpath("rasters/Marma_DEM_2016.tif").__str__(), conform_to=dems[reference_grid_year])
        if year == 2021:
            return load_dem(gis_data_path.joinpath("rasters/marma_DEM_2021_20210810.tif").__str__(), conform_to=dems[reference_grid_year])

    dems[reference_grid_year] = load_one_dem(reference_grid_year)

    dems.update({year: load_one_dem(year) for year in filter(lambda y: y != reference_grid_year, available_years)})

    unstable_terrain = load_snow_and_ice(
        gis_data_path.joinpath("shapes/glacier.geojson").__str__(),
        gis_data_path.joinpath("shapes/snow.geojson").__str__(),
    )
    return dems, unstable_terrain




def test_parse_txt_dem():

    marm1959_path = "GIS/rasters/marm195910m.txt"

    dem = parse_txt_dem(marm1959_path)

    assert dem.res == (10.0, 10.0)
    assert dem.crs.to_epsg() == 3006
