from __future__ import annotations

import contextlib
import datetime
import io
import warnings
import tempfile
from pathlib import Path

import autosklearn.regression
import geoutils as gu
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.stats
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
import xdem
from numpy.lib import stride_tricks
from tqdm import tqdm

import marma
import marma.inputs
import marma.main
import marma.mb_parsing


def generate_ddems(dems: dict[int, xdem.DEM], max_interpolation_distance: float = 200) -> list[xdem.dDEM]:


    ddems: list[xdem.dDEM] = []

    years = sorted(list(dems.keys()))

    for i in range(1, len(years)):
        ddem = xdem.dDEM(
            dems[years[i]] - dems[years[i - 1]],
            start_time=dems[years[i - 1]].date,
            end_time=dems[years[i]].date,
        )
        ddem.data = xdem.volume.linear_interpolation(
            ddem.data, max_search_distance=int(max_interpolation_distance / ddem.res[0])
        )

        ddems.append(ddem)

    return ddems


def coregister(
    dems: dict[int, xdem.DEM], reference_year: int, stable_terrain_masks: dict[int, np.ndarray]
) -> dict[int, xdem.DEM]:

    dems_coreg = {
        year: xdem.coreg.NuthKaab()
        .fit(dems[reference_year], dems[year], inlier_mask=stable_terrain_masks[year])
        .apply(dems[year])
        if year != reference_year
        else dems[reference_year]
        for year in dems
    }

    for year in dems:
        dems_coreg[year].date = dems[year].date
    return dems_coreg


def get_slope_and_max_curvature(dem: xdem.DEM) -> tuple[gu.Raster, gu.Raster]:

    slope, planform_curvature, profile_curvature = xdem.terrain.get_terrain_attribute(
        dem, ["slope", "planform_curvature", "profile_curvature"]
    )

    maxc = gu.Raster.from_array(
        np.maximum(np.abs(planform_curvature.data), np.abs(profile_curvature.data)),
        transform=dem.transform,
        crs=dem.crs,
        nodata=-9999,
    )

    return slope, maxc


@numba.njit(parallel=True)
def nmad_filter(array: np.ndarray, kernel_size: int, subsample: int = 0) -> np.ndarray:
    """
    Calculate the local NMAD of each pixel and its neighbours within the given kernel size.

    """
    # Initialize an output array with nans
    nmad_values = np.zeros(array.size, dtype="float32") + np.nan

    # If subsample == 0, use the whole array, otherwise a random subset of it.
    indices_to_use = np.arange(array.size) if subsample == 0 else np.random.choice(array.size, size=subsample)

    # TODO: Make sure this is parallel
    # Loop over each array index and calculate the NMAD of it and its neighbours
    for i in indices_to_use:
        row = int(i // array.shape[1])
        col = i - row * array.shape[1]

        # Extract the neighbouring area of the pixel (stopping at the borders)
        subset = array[
            max(0, row - kernel_size) : min(array.shape[0] - 1, row + kernel_size),
            max(0, col - kernel_size) : min(array.shape[1] - 1, col + kernel_size),
        ]

        # If the subset was for some reason empty or if it contains only nans, skip it.
        if subset.size == 0 or np.all(~np.isfinite(subset)):
            continue

        # Append the calculated NMAD to the right spot in the array
        nmad_values[i] = 1.4826 * np.nanmedian(np.abs(subset - np.nanmedian(subset)))

    return nmad_values


@numba.njit(parallel=True)
def mean_filter(array: np.ndarray, kernel_size: int, subsample: int = 0, nan_threshold: float = 0.8) -> np.ndarray:
    """
    Calculate the local NMAD of each pixel and its neighbours within the given kernel size.

    """
    # Initialize an output array with nans
    mean_values = np.zeros(array.size, dtype="float32") + np.nan

    # If subsample == 0, use the whole array, otherwise a random subset of it.
    indices_to_use = np.arange(array.size) if subsample == 0 else np.random.choice(array.size, size=subsample)

    # TODO: Make sure this is parallel
    # Loop over each array index and calculate the NMAD of it and its neighbours
    for i in indices_to_use:
        row = int(i // array.shape[1])
        col = i - row * array.shape[1]

        # Extract the neighbouring area of the pixel (stopping at the borders)
        subset = array[
            max(0, row - kernel_size) : min(array.shape[0] - 1, row + kernel_size),
            max(0, col - kernel_size) : min(array.shape[1] - 1, col + kernel_size),
        ]

        # If the subset was for some reason empty or if it contains only nans, skip it.
        if subset.size == 0 or (np.count_nonzero(~np.isfinite(subset)) / subset.size) < nan_threshold:
            continue

        # Append the calculated NMAD to the right spot in the array
        mean_values[i] = np.nanmean(subset)

    return mean_values


def get_spatially_correlated_nmad(
    ddem: xdem.dDEM,
    stable_terrain_mask: np.ndarray,
    steps: int = 25,
    step_feature_count: int = 5000,
    progress: bool = True,
):

    norm_dh = np.where(~stable_terrain_mask, np.nan, ddem.data.filled(np.nan)).squeeze()
    if ddem.error is not None:
        # Normalize by the error (we only want the spatially correlated error, not the terrain-correlated error)
        norm_dh /= ddem.error.squeeze()

    assert np.any(np.isfinite(norm_dh))

    # Remove extreme outliers as this may mess with the result.
    norm_dh[np.abs(norm_dh) > (4 * xdem.spatialstats.nmad(norm_dh))] = np.nan

    # The full area of the raster (the pixel count times the pixel area)
    scene_area = np.prod(ddem.shape) * np.prod(ddem.res)

    # Generate areas to test (from 0.01 to half an order of magnitude below the full scene's area)
    areas = 10.0 ** np.linspace(-1, np.log10(scene_area) - 0.5, steps)

    # Remove areas that are smaller than five pixels (because each cell needs some friends for a good NMAD)
    # areas = areas[(areas > np.prod(ddem.res) * 5)]

    # The kernel sizes will be the pixel width of the area rectangles
    kernel_sizes = (np.sqrt(areas) / np.mean(ddem.res)).astype(int)
    # The number of samples will be 4 times amount of rectangles that can fit in the scene.
    # It is clipped to a minimum of 200 samples and a maximum of 5000 samples.
    n_samples = np.clip(4 * scene_area / areas, 200, step_feature_count).astype(int)

    variance = pd.DataFrame(dtype=float)
    variance.index.name = "area"
    full_nmad = xdem.spatialstats.nmad(norm_dh)

    progress_bar = tqdm(total=n_samples.sum(), disable=not progress)
    for i, area in enumerate(areas):

        # Run the NMAD filter with the associated kernel size and number of samples for this area.
        means = mean_filter(norm_dh, kernel_sizes[i], subsample=n_samples[i])
        # Remove all pixels that didn't have a value from the start.
        means[~np.isfinite(norm_dh.ravel())] = np.nan

        if np.all(~np.isfinite(means)):
            continue

        nmad = xdem.spatialstats.nmad(means)

        finite_means = means[np.isfinite(means)]
        np.random.shuffle(finite_means)

        nmads_sub = np.array(
            [
                xdem.spatialstats.nmad(arr)
                for arr in np.array_split(finite_means, max(1, int(finite_means.size * 0.04)))
                if np.any(np.isfinite(arr))
            ]
        )

        # For this area, add the median NMAD, the standard deviation of the different samples, the sample count,
        # the approximate radius of the rectangle (the mean of the smallest and longest distance to the edge),
        # and the kernel size (in pixels) that was used.
        variance.loc[area, ["nmad", "std", "lower_quartile", "upper_quartile", "count", "radius", "kernel_size"]] = (
            nmad,
            np.nanstd(means),
            np.nanpercentile(nmads_sub, 25),
            np.nanpercentile(nmads_sub, 75),
            np.count_nonzero(np.isfinite(means)),
            np.mean([area ** 0.5, area ** 0.5 * (2 ** 0.5)]),
            kernel_sizes[i],
        )

        progress_bar.update(n_samples[i])

    progress_bar.close()

    # Add the variance of the full scene in the end (no fancy looping has to be made here)
    """
    variance.loc[scene_area, ["nmad", "std", "upper_quartile", "lower_quartile", "count", "radius", "kernel_size"]] = (
        full_nmad,
        0,
        full_nmad,
        full_nmad,
        np.count_nonzero(np.isfinite(norm_dh)),
        np.mean([scene_area ** 0.5, scene_area ** 0.5 * (2 ** 0.5)]),
        np.mean(norm_dh.shape, dtype=int),
    )
    """

    # Some area steps may have sampled the exact same values. These should be removed.
    # We rely on the uniqueness of the NMAD value to filter these out.
    variance = variance.drop_duplicates(subset=["nmad"])

    # Set the area in pixel coordinates.
    variance["area_px"] = variance.index / np.prod(ddem.res)

    return variance


def error(
    ddems: list[xdem.dDEM],
    slope: gu.Raster,
    max_curvature: gu.Raster,
    stable_terrain_masks: dict[int, np.ndarray],
    variance_steps: int = 25,
    variance_feature_count: int = 5000,
) -> list[xdem.dDEM]:

    for i, ddem in tqdm(enumerate(ddems), total=len(ddems)):
        stable_terrain_mask = stable_terrain_masks[ddem.start_time.year]

        ddem.stable_terrain_mask = stable_terrain_mask

        ddem_arr = ddem.data.copy()
        inlier_mask = stable_terrain_mask & np.isfinite(ddem_arr.filled(np.nan))
        ddem_arr = ddem_arr[inlier_mask]
        slope_arr = slope.data[inlier_mask]
        maxc_arr = max_curvature.data[inlier_mask]

        custom_bins = [
            np.unique(
                np.concatenate(
                    [
                        np.quantile(arr, np.linspace(start, stop, num))
                        for start, stop, num in [(0, 0.95, 20), (0.96, 0.99, 5), (0.991, 1, 10)]
                    ]
                )
            )
            for arr in [slope_arr, maxc_arr]
        ]

        error_model = xdem.spatialstats.interp_nd_binning(
            xdem.spatialstats.nd_binning(
                values=ddem_arr,
                list_var=[slope_arr, maxc_arr],
                list_var_names=["slope", "maxc"],
                list_var_bins=custom_bins,
                statistics=["count", xdem.spatialstats.nmad],
            ),
            list_var_names=["slope", "maxc"],
            min_count=30,
        )
        ddem.error = error_model((slope.data, max_curvature.data)).reshape(slope.data.shape)

        year_label = f"{ddem.start_time.year} to {ddem.end_time.year}"

        # Standardize by the error, remove snow/ice values, and remove large outliers.
        standardized_dh = np.where(~stable_terrain_mask, np.nan, ddem.data / ddem.error)
        standardized_dh[np.abs(standardized_dh) > (4 * xdem.spatialstats.nmad(standardized_dh))] = np.nan

        standardized_std = np.nanstd(standardized_dh)

        norm_dh = standardized_dh / standardized_std

        # This may fail due to the randomness of the analysis, so try to run this five times
        for i in range(5):
            try:
                variogram = xdem.spatialstats.sample_empirical_variogram(
                    values=norm_dh.squeeze(),
                    gsd=ddem.res[0],
                    subsample=50,
                    n_variograms=10,
                    runs=30,
                )
            except ValueError as exception:
                if i == 4:
                    raise exception
                continue
            break

        vgm_model, params = xdem.spatialstats.fit_sum_model_variogram(["Sph", "Sph"], variogram)

        variance = get_spatially_correlated_nmad(
            ddem, stable_terrain_mask, progress=False, steps=variance_steps, step_feature_count=variance_feature_count
        )

        ddem.variograms = {
            "empirical_variance": variance,
            "vgm_model": vgm_model,
            "vgm_params": params,
            "standardized_std": standardized_std,
            "variogram": variogram,
        }

    return ddems


def get_effective_samples(ddems: list[xdem.dDEM], outlines: gu.Vector):

    for ddem in ddems:

        dissolved_outline = (
            outlines.query(f"(name == 'Marma') & ((year == {ddem.start_time.year}) | (year == {ddem.end_time.year}))")
            .ds.dissolve()
            .area.sum()
        )

        neff = xdem.spatialstats.neff_circ(
            dissolved_outline,
            [
                (ddem.variograms["vgm_params"][0], "Sph", ddem.variograms["vgm_params"][1]),
                (ddem.variograms["vgm_params"][2], "Sph", ddem.variograms["vgm_params"][3]),
            ],
        )

        ddem.variograms["n_effective_samples"] = neff


def volume_change(ddems: list[xdem.dDEM], outlines: gu.Vector):

    vol_change_data = {}
    for ddem in ddems:
        queried_outlines = outlines.query(
            f"(name == 'Marma') & ((year == {ddem.start_time.year}) | (year == {ddem.end_time.year}))"
        )
        glacier_mask = queried_outlines.create_mask(ddem)

        """
        error_model = scipy.interpolate.interp1d(
            ddem.variograms["empirical_variance"].index,
            ddem.variograms["empirical_variance"]["nmad"],
            fill_value="extrapolate",
        )
        """

        merged_area = queried_outlines.ds.dissolve().area.sum()

        vol_change_data[pd.Interval(pd.Timestamp(ddem.start_time), pd.Timestamp(ddem.end_time))] = {
            "mean_dh": np.nanmean(ddem.data[glacier_mask]),
            "dh_error": np.nanmean(ddem.error[glacier_mask]) / np.sqrt(ddem.variograms["n_effective_samples"]),
            "merged_area": merged_area,
            "start_area": queried_outlines.ds[queried_outlines.ds["year"] == ddem.start_time.year].area.sum(),
            "end_area": queried_outlines.ds[queried_outlines.ds["year"] == ddem.end_time.year].area.sum(),
            "pixel_count": np.count_nonzero(glacier_mask),
        }
    vol_change = pd.DataFrame(vol_change_data).T

    # vol_change["dh_error"] = vol_change["topographic_error"] + vol_change["spatially_correlated_error"]
    vol_change["mean_dv"] = vol_change["mean_dh"] * vol_change["merged_area"]
    vol_change["dv_error"] = vol_change["dh_error"] * vol_change["merged_area"]

    return vol_change


class InterpolatedDEM:
    def __init__(self, dems: dict[int, xdem.DEM]):
        self.dems = dems

    def interpolate(self, year: int | float) -> np.ndarray:
        if year in self.dems:
            return self.dems[year].data.squeeze()


        years = np.array(list(self.dems.keys())).astype(float)

        if year > years.max():
            return self.dems[years.max()].data.squeeze()

        lower_year = years[years < year].max()
        upper_year = years[years > year].min()

        relative_weight = (year - lower_year) / (upper_year - lower_year)

        lower_data = self.dems[int(lower_year)].data.squeeze()
        upper_data = self.dems[int(upper_year)].data.squeeze()

        valid_values = np.isfinite(lower_data).astype(int) & np.isfinite(upper_data).astype(int)

        data = np.where(
            valid_values,
            np.average([lower_data, upper_data], axis=0, weights=[1 - relative_weight, relative_weight]),
            np.nansum([lower_data, upper_data], axis=0),
        )

        return data

    def sample(self, easting: np.ndarray, northing: np.ndarray, year: int | float) -> np.ndarray:
        rows, cols = [arr.round(0).astype(int) for arr in self.dems[min(self.dems)].xy2ij(easting, northing)]

        return self.interpolate(year)[rows, cols]


def snow_probings():
    probings, stakes, densities = marma.mb_parsing.read_all_data()

    storgl = pd.read_csv(
        "input/SITES_GL-MB_TRS_SGL_1946-2020_L2_annual.csv", skiprows=22, index_col="TIMESTAMP"
    ).astype(float)

    probings["year"] = probings["date"].apply(lambda d: d.year)
    densities["year"] = densities["date"].apply(lambda d: d.year)

    probings["x"] = probings.geometry.x
    probings["y"] = probings.geometry.y

    dems, _, _ = marma.main.prepare_dems(2016)

    tdem = InterpolatedDEM(dems)
    

    attributes = ["slope", "planform_curvature", "profile_curvature"]


    for year, data in tqdm(probings.groupby("year", as_index=False), desc="Preparing data"):

        yearly_density = densities[densities["year"] == year]
        if yearly_density.shape[0] == 0:
            yearly_density = densities

        mean_density = np.average(
            yearly_density["density"].astype(float), weights=yearly_density["depth_cm"].apply(lambda i: i.length)
        )
        probings.loc[data.index, "snow_depth_we"] = data["snow_depth_cm"] * mean_density

        dem = tdem.interpolate(year)

        rows, cols = [arr.round(0).astype(int) for arr in dems[2016].xy2ij(data["geometry"].x, data["geometry"].y)]
        probings.loc[data.index, "storgl_wb"] = storgl.loc[year, "MB_W"]

        probings.loc[data.index, "dem"] = dem[rows, cols]

        attribute_data = dict(zip(attributes, xdem.terrain.get_terrain_attribute(dem, resolution=dems[2016].res[0], attribute=attributes)))

        for attribute in attributes:
            probings.loc[data.index, attribute] = attribute_data[attribute][rows, cols]

        probings["snow_depth_we"] = probings["snow_depth_we"].astype(float)

    train_cols = attributes + ["dem", "storgl_wb"]
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        probings[train_cols].values, probings["snow_depth_we"].values, random_state=1
    )

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        n_jobs=-1,
    )

    automl.fit(x_train, y_train, dataset_name="snow_depth")

    train_predictions = automl.predict(x_train)
    print("Train R2 score:", sklearn.metrics.r2_score(y_train, train_predictions))
    test_predictions = automl.predict(x_test)
    print("Test R2 score:", sklearn.metrics.r2_score(y_test, test_predictions))

    for year in storgl.index:
        if year < 1959:
            continue
        # xcoords, ycoords = dem.coords(offset="center")
        storgl_wb = [storgl.loc[year, "MB_W"]] * dem.size
        dem = tdem.interpolate(year)

        attribute_data = dict(zip(attributes, xdem.terrain.get_terrain_attribute(dem, resolution=dems[2016].res[0], attribute=attributes)))

        x_values = [attribute_data[key].ravel() for key in attributes] + [dem.ravel(), storgl_wb]

        prediction = automl.predict(np.transpose(x_values)).reshape(dem.shape)

        plt.imshow(prediction)
        plt.show()

    # plt.scatter(probings["snow_depth_norm"], predictions, alpha=0.3)
    # plt.show()

def ablation():
    probings, stakes, densities = marma.mb_parsing.read_all_data(Path("input/massbalance"))

    storgl = pd.read_csv(
        "input/SITES_GL-MB_TRS_SGL_1946-2020_L2_annual.csv", skiprows=22, index_col="TIMESTAMP"
    ).astype(float)

    for data in [probings, densities, stakes]:
        data["year"] = data["date"].apply(lambda d: d.year)

    print(probings[probings["year"] == 2008])

    mean_densities = densities.groupby("year").apply(lambda df: np.average(df["density"].astype(float), weights=df["depth_cm"].apply(lambda i: i.length)))

    mean_densities[2010] = mean_densities.mean()


