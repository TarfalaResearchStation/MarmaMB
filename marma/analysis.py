from __future__ import annotations

import contextlib
import datetime
import io
import warnings

import geoutils as gu
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import scipy.interpolate
import xdem
from numpy.lib import stride_tricks
from tqdm import tqdm


def generate_ddems(dems: dict[int, xdem.DEM], max_interpolation_distance: float = 200) -> list[xdem.dDEM]:

    ddems: list[xdem.dDEM] = []

    years = sorted(list(dems.keys()))

    for i in range(1, len(years)):
        ddem = xdem.dDEM(
            dems[years[i]] - dems[years[i - 1]],
            start_time=datetime.date(years[i - 1], 8, 1),
            end_time=datetime.date(years[i], 8, 1),
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
    norm_dh[np.abs(norm_dh) > (4 * xdem.spatial_tools.nmad(norm_dh))] = np.nan

    # The full area of the raster (the pixel count times the pixel area)
    scene_area = np.prod(ddem.shape) * np.prod(ddem.res)

    # Generate areas to test (from 0.01 to half an order of magnitude below the full scene's area)
    areas = 10.0 ** np.linspace(-1, np.log10(scene_area) - 0.5, steps)

    # Remove areas that are smaller than five pixels (because each cell needs some friends for a good NMAD)
    #areas = areas[(areas > np.prod(ddem.res) * 5)]

    # The kernel sizes will be the pixel width of the area rectangles
    kernel_sizes = (np.sqrt(areas) / np.mean(ddem.res)).astype(int)
    # The number of samples will be 4 times amount of rectangles that can fit in the scene.
    # It is clipped to a minimum of 200 samples and a maximum of 5000 samples.
    n_samples = np.clip(4 * scene_area / areas, 200, step_feature_count).astype(int)

    variance = pd.DataFrame(dtype=float)
    variance.index.name = "area"
    full_nmad = xdem.spatial_tools.nmad(norm_dh)

    progress_bar = tqdm(total=n_samples.sum(), disable=not progress)
    for i, area in enumerate(areas):

        # Run the NMAD filter with the associated kernel size and number of samples for this area.
        means = mean_filter(norm_dh, kernel_sizes[i], subsample=n_samples[i])
        # Remove all pixels that didn't have a value from the start.
        means[~np.isfinite(norm_dh.ravel())] = np.nan

        if np.all(~np.isfinite(means)):
            continue

        nmad = xdem.spatial_tools.nmad(means)

        finite_means = means[np.isfinite(means)]
        np.random.shuffle(finite_means)

        nmads_sub = np.array(
            [
                xdem.spatial_tools.nmad(arr)
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
                statistics=["count", xdem.spatial_tools.nmad],
            ),
            list_var_names=["slope", "maxc"],
            min_count=30,
        )
        ddem.error = error_model((slope.data, max_curvature.data)).reshape(slope.data.shape)

        year_label = f"{ddem.start_time.year} to {ddem.end_time.year}"

        # Standardize by the error, remove snow/ice values, and remove large outliers.
        standardized_dh = np.where(~stable_terrain_mask, np.nan, ddem.data / ddem.error)
        standardized_dh[np.abs(standardized_dh) > (4 * xdem.spatial_tools.nmad(standardized_dh))] = np.nan

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

    #vol_change["dh_error"] = vol_change["topographic_error"] + vol_change["spatially_correlated_error"]
    vol_change["mean_dv"] = vol_change["mean_dh"] * vol_change["merged_area"]
    vol_change["dv_error"] = vol_change["dh_error"] * vol_change["merged_area"]

    return vol_change
