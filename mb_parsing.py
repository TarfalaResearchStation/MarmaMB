"""
# Field data document parsers for Mårmaglaciären.
Parser functions for every year since 1990 exist here, to collect and standardize the data.
Almost every year has a different template (or a variation thereof), so many custom parsers were required.

Most of the original data exist.
Late-summer stake readings from 2003 and 2011 are mysteriously gone, but calculated ablation values exist.
Thus, the original late-summer height can be back-tracked.
Density pits were not dug in 2010, and data are therefore missing from that year.
Finally, many years are missing the date at which the readings were taken.
They are filled in with a "most-likely" standard date and are noted appropriately.

The parsers themselves are mostly devoid of intepretation, unless it is explicitly mentioned.
Some missing pieces of information are filled, such as:
    - Snow depth == 0 in August -> probably ice.
    - Density pits of equal depth as the probing depth at the top stake -> it was probably dug at the top stake.
    - A stake without a coordinate but with the same ID as one at the previous year -> they are at the same place.
    - A snow probing "taken in September" among others taken in May -> the date is probably wrong.

Author:
    Erik Mannerfelt
Date:
    September 2021
"""
from __future__ import annotations

import datetime
import io
import json
import os
import re
import warnings
from pathlib import Path
import tarfile
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd

__version__ = "0.0.1"

SURFACE_TYPES = ["s", "i", "ns", "si"]


def standard_summer_date(year: int) -> datetime.date:
    """In case of a missing summer date, use this date as a placeholder."""
    return datetime.date(int(year), 9, 1)


def standard_winter_date(year: int) -> datetime.date:
    """In case of a missing winter date, use this date as a placeholder."""
    return datetime.date(int(year), 5, 1)


def read_all_data(base_directory: Path):
    """
    Read the data from all years in the 'base_directory'.

    The parser for each year is guessed based on the data it has.

    :param base_directory: The directory that should contain year-directories (containing the data)
    """
    # Find all directories in the 'base_directory' that are numeric (thus assumed to represent years)
    year_dirs = filter(lambda d: d.isnumeric(), os.listdir(base_directory))

    probing_dfs = []
    stake_dfs = []
    density_dfs = []
    # Loop over all years and parse the data
    for year in sorted(year_dirs, reverse=True):
        all_files = os.listdir(base_directory.joinpath(year))

        # Find all excel files and all text files (with expected extensions)
        xls_files = list(filter(lambda s: "xls" in s and "lock" not in s, all_files))
        txt_files = list(
            filter(lambda s: any(ext in s for ext in ["txt", "csv", "geojson"]) and "lock" not in s, all_files)
        )
        # If for any reason the directory is empty, skip it.
        if len(xls_files) == 0 and len(txt_files) == 0:
            continue

        # Instantiate the data variable which will contain all file contents
        data: dict[str, pd.DataFrame | list[str]] = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Data Validation extension is not supported")
            i = 0
            # Read each excel file
            for filename in xls_files:
                for key, value in pd.read_excel(
                    base_directory.joinpath(year).joinpath(filename), sheet_name=None, header=None
                ).items():
                    key_to_use = key
                    # If the key already exists (e.g. 'Sheet1'), make a new unique name
                    for i in range(1, 100):
                        if key_to_use in data:
                            key_to_use = key + f"_{i}"
                        else:
                            break

                    data[key_to_use] = value

            # Read all text files and split the lines.
            for filename in txt_files:
                with open(base_directory.joinpath(year).joinpath(filename), encoding="utf-8") as infile:
                    data[os.path.splitext(filename)[0]] = infile.read().splitlines()

        # Figure out the parser version and parse the data.
        probings, stakes, densities = figure_out_version(data, int(year))

        # Some columns were duplicated at some point, so these need to be removed.
        # TODO: Fix the parsers instead!
        stakes = stakes.loc[:, ~stakes.columns.duplicated()]

        probing_dfs.append(probings)
        stake_dfs.append(stakes)
        density_dfs.append(densities)

    # Concatenate the results and keep only the columns of interest.
    densities = pd.concat(density_dfs, ignore_index=True)[
        ["depth_cm", "density", "date", "parser", "location", "note", "profile_id", "geometry"]
    ]
    densities["density"] = densities["density"].astype(float)
    probings = pd.concat(probing_dfs, ignore_index=True)[["stake_id", "snow_depth_cm", "date", "geometry", "parser"]]

    stakes = pd.concat(stake_dfs, ignore_index=True)[
        ["stake_id", "stake_height_cm", "snow_depth_cm", "note", "surface", "date", "geometry", "parser"]
    ]

    # Stakes where no snow depth was reported is assumed to be 0
    stakes["snow_depth_cm"] = stakes["snow_depth_cm"].fillna(0)

    # For some reason, the "nodata" version of a point is Point(inf, inf), which makes for really funny plots.
    stakes.loc[stakes["geometry"].apply(lambda p: ~np.isfinite(p.x) if p is not None else True), "geometry"] = pd.NA

    # For all stakes and probings, if they are missing a coordinate but have an associated stake_id,
    # try to find another measurement with the same stake_id that has a coordinate, and assign this.
    for dataframe in [stakes, probings]:
        for i, point in dataframe.loc[dataframe["geometry"].isna()].iterrows():
            other_measurements = pd.concat(
                [
                    df.loc[(df["stake_id"] == point["stake_id"]) & (~df["geometry"].isna()), "geometry"]
                    for df in [stakes, probings]
                ]
            )
            if np.size(other_measurements) == 0:
                continue
            dataframe.loc[i, "geometry"] = other_measurements.iloc[0]

    # Filter the stakes and probings by whether measurements exist or not.
    # TODO: Fix the parsers instead!
    stakes = stakes[
        is_number(stakes["stake_height_cm"]) & (is_number(stakes["snow_depth_cm"]) | stakes["snow_depth_cm"].isna())
    ]
    probings = probings[(~probings["geometry"].isna()) & (is_number(probings["snow_depth_cm"]))]

    stakes.loc[stakes["surface"].apply(lambda s: isinstance(s, str) and len(s.strip()) == 0), "surface"] = None

    surface_translations = {"nysnö": "ns", "snow": "s", "ice": "i"}
    stakes["surface"] = stakes["surface"].str.lower()
    stakes["surface"] = stakes["surface"].apply(
        lambda s: s if s not in surface_translations else surface_translations[s]
    )

    # Convert all datetimes to dates (times sometimes exist, but not always, and the timezone is unknown)
    for dataframe in [stakes, probings]:
        dataframe["date"] = dataframe["date"].apply(lambda d: d if isinstance(d, datetime.date) else d.date())

    # Make sure the stakes and probings are proper GeoDataFrames
    stakes = gpd.GeoDataFrame(stakes, crs=3006)
    probings = gpd.GeoDataFrame(probings, crs=3006).sort_values("date")

    validate_data(probings, stakes, densities)
    interpret_data(probings, stakes, densities)

    return probings, stakes, densities


def validate_data(probings: gpd.GeoDataFrame, stakes: gpd.GeoDataFrame, densities: pd.DataFrame) -> None:
    """Validate the probing, stake and density data."""

    # Loop over all dataframes and check for data gaps. Then do individual validations.
    for dataframe, name in [(probings, "probings"), (stakes, "stakes"), (densities, "densities")]:
        previous_year = None
        # Check for data gaps. There is a known 2010 data gap in the density series.
        for year in sorted(dataframe["date"].apply(lambda d: d.year).unique()):
            if all(
                [previous_year is not None, (year - (previous_year or 0)) > 1, year != 2010 and name != "densities"]
            ):
                raise ValueError(f"Year gap for {name} between {previous_year} and {year}")
            previous_year = year

            data = dataframe.loc[dataframe["date"].apply(lambda d: d.year) == year]

            """
            if name == "densities":
                # Check that the amount of pits correspond to the amount of unique pit identifiers
                left = data["depth_cm"].apply(lambda i: i.left)
                # The amount of pit tops is when the count of the minimum depth values (most often 0 cm)
                n_starts = left[left == left.min()].shape[0]
                # The amount of locations is the unique count of location, id and geometry combinations
                n_locations = data[["location", "profile_id", "geometry"]].astype(str).sum(axis=1).unique().shape[0]

                # If this doesn't match, a label is missing
                if n_starts != n_locations:
                    raise ValueError(f"Unequal pit starts ({n_starts}) compared to unique locations ({n_locations}): {data}")
            """

            # Validate that all stakes and probings have coordinates
            if name in ["stakes", "probings"]:
                if not np.all(np.isfinite(data.geometry.x)):
                    raise ValueError(f"Missing geometry columns:\n{data[~np.isfinite(data.geometry.x)]}")
            if name == "stakes":
                # Validate that no unexpected surface names exist.
                surfaces = data["surface"].unique()
                for surface in surfaces:
                    if pd.isna(surface):
                        continue
                    if surface not in ["s", "ns", "i", "si"]:
                        raise ValueError(f"Unknown surface type: {surface}")

                # Check that the theoretical minimum of two summer and winter stake readings exist.
                winter_measurements = data.loc[data["date"].apply(lambda d: d.month < 6)]
                summer_measurements = data.loc[data["date"].apply(lambda d: d.month > 6)]
                # At least two stakes need to have both winter and summer measuements (hence the overlapping)
                overlapping = (
                    summer_measurements.set_index("stake_id")["stake_height_cm"]
                    - winter_measurements.set_index("stake_id")["stake_height_cm"]
                )
                overlapping = overlapping[~overlapping.isna()]
                for subset_data, subset in [
                    (winter_measurements, "winter"),
                    (summer_measurements, "summer"),
                    (overlapping, "overlapping"),
                ]:
                    if subset_data.shape[0] < 2:
                        raise ValueError(f"{year} has too few {subset} stakes: {subset_data.shape[0]}")

                # A 6 m difference or a negative difference (winter higher than summer) is impossible/improbable
                if any(overlapping.abs()) > 6 or any(overlapping < 0.0):
                    raise ValueError(f"Unreasonable stake dfference:\n{overlapping}")


def interpret_data(probings: gpd.GeoDataFrame, stakes: gpd.GeoDataFrame, densities: pd.DataFrame) -> None:
    """Modify the dataframes inplace and add data that are reasonable."""
    # Fill in missing stake heights, snow depths, and surface types.
    for i, stake in stakes.iterrows():
        # If no valid stake height exists, it's most likely 6m
        if not is_number(stake["stake_height_cm"]):
            stakes.loc[i, "stake_height_cm"] = 600

        # For winter data
        if stake["date"].month < 6:
            # If no surface info exists but there is snow, set the surface as snow
            if stake["surface"] not in SURFACE_TYPES and float(stake["snow_depth_cm"]) > 0:
                stakes.loc[i, "surface"] = "s"
        # For summer data
        elif stake["date"].month >= 6:
            # If the stake doesn't have an associated snow depth, it's probably bare ice
            if not is_number(stake["snow_depth_cm"]):
                stakes.loc[i, "snow_depth_cm"] = 0
            if stake["surface"] not in SURFACE_TYPES:
                # If no surface info exists but there is snow, set the surface as snow
                if float(stake["snow_depth_cm"]) > 0:
                    stakes.loc[i, "surface"] = "s"
                # OtherwICE set it as ice
                else:
                    stakes.loc[i, "surface"] = "i"

    densities.loc[densities["note"].isna(), "note"] = ""

    # Associate coordinates with all density pits.
    for year in densities["date"].apply(lambda d: d.year).unique():
        year_data = densities.loc[densities["date"].apply(lambda d: d.year) == year]
        # Extract the winter stake readings from the same year.
        year_stakes = stakes.loc[stakes["date"].apply(lambda d: d.year == year and d.month < 6)].copy()
        # If the pit(s) already have coordinates, go to the next one
        if (~year_data["geometry"].isna()).all():
            continue

        # Sort the stakes from west to east (westernmost is assumed to be farthest up)
        year_stakes = year_stakes.iloc[np.argsort([geom.x for geom in year_stakes.geometry.values])]

        # If there is only one pit, assume that it is taken from the highest measured stake
        if np.unique(year_data[["location", "profile_id"]].astype(str).sum(axis=1)).shape[0] == 1:
            densities.loc[year_data.index, "geometry"] = repeat_geometry(
                year_stakes["geometry"].iloc[0], year_data.shape[0]
            )
            densities.loc[
                year_data.index, "note"
            ] += " Location assumed to be at the stake highest in the accumulation area."
            continue

        # If there are multiple pits and they have locations corresponding to stake names, use those stakes
        if any(loc in year_stakes["stake_id"].values for loc in year_data["location"]):
            for i, stake in year_stakes.iterrows():
                overlapping = densities["location"] == stake["stake_id"]
                if np.count_nonzero(overlapping) == 0:
                    continue
                densities.loc[overlapping, "geometry"] = repeat_geometry(
                    stake["geometry"], np.count_nonzero(overlapping)
                )

        # At this point, all single pits and those with associated stake names have gotten coordinates.
        # Now, there will only be one left (based on visual inspection), and will be assigned to the highest stake.
        # It didn't seem like the year_data was updated by the last call, so make a new view
        year_data = densities.loc[densities["date"].apply(lambda d: d.year) == year]

        nans = year_data[year_data["geometry"].isna()].index
        densities.loc[nans, "geometry"] = repeat_geometry(year_stakes["geometry"].iloc[0], nans.shape[0])
        densities.loc[nans, "note"] += " Location assumed to be at the stake highest in the accumulation area."

    if ((densities["geometry"]).isna()).any():
        raise NotImplementedError(densities)


def repeat_geometry(geometry: object, repeats: int) -> np.ndarray:
    """
    Repeat the given geometry object N times.

    This is needed as e.g. np.array([geom] * 10) triggers warnings about not respecting the __array_interface__ of geom.

    :param geometry: Any object to be repeated.
    :param repeats: The number of repeats.

    :returns: A numpy array with the repeated objects.
    """
    geom_arr = np.empty(repeats, dtype=object)
    for i in range(geom_arr.shape[0]):
        geom_arr[i] = geometry

    return geom_arr


def figure_out_version(
    data: dict[str, pd.DataFrame | list[str]], year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the given data by figuring out the proper parser version, and then calling that parser.

    :param data: The data to parse.
    :param year: Which year the data is associated with.

    :raises ValueError: If the data did not match any pre-set parser pattern.
    :raises Exception: From the associated parser.

    :returns: The probings, stakes and densities of that year.
    """
    try:
        # The 1998 version is checked first as it may otherwise be caught by the 1999 (or was it 2001?) version
        if all(key in data for key in ["densities_digitized", "probings_digitized", "stakes_digitized"]):
            return parse_1998_version(data, year)
        if all(key in data for key in ["Mårma_DENSITY", "Mårma_STAKES", "Mårma_PROBE"]):
            return parse_2019_version(data, year)
        if all(key in data for key in ["Mårma Densitet", "Mårma stakes", "Mårma Sondering"]):
            return parse_2017_version(data, year)
        if all(key in data for key in ["Density", "Stake ablation", "Probing data"]):
            return parse_2016_version(data, year)
        if all(key in data for key in ["Density", "Summer balance calc", "Probing data", "ablation"]):
            return parse_2011_version(data, year)
        if all(key in data for key in ["Density", "Summer balance calc", "Probing data"]):
            return parse_2013_version(data, year)
        if all(key in data for key in ["Density", "Stakar", "sond"]):
            return parse_2005_version(data, year)
        if all(key in data for key in ["Rådata", "Stakborr"]):
            return parse_2004_version(data, year)
        if all(key in data for key in ["sondering", "stake_koord"]):
            return parse_2003_version(data, year)
        if all(key in data for key in ["marma2002"]):
            return parse_2002_version(data, year)
        if all(key in data for key in ["stake_koord", "Sond_koord"]):
            return parse_2001_version(data, year)
        if all(key in data for key in ["Graf", "Beräkningar"]):
            return parse_1999_version(data, year)

    except Exception as exception:
        print(f"Failed on {year=}")
        raise exception

    raise ValueError(f"Data sheet for {year=} did not match patterns:\n{data}")


def parse_1998_version(
    data: dict[str, pd.DataFrame | list[str]], year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the "1998" data format.

    The 1998 data format is made of two geojson files (stakes and probings) and one csv (density pits).
    This format is made from a re-digitization of original data by Erik Mannerfelt in 2021.

    The version year has nothing to do with when it was created;
    it is the first year it was encountered going back in time.
    For example, 1997 may use the 1998 parser, but 1999 will not use the 1998 parser.

    :param data: The data to parse.
    :param year: Which year the data is associated with.

    :returns: The probings, stakes and densities of that year.
    """
    # Load the probings geojson
    probings = gpd.GeoDataFrame.from_features(json.loads("\n".join(data["probings_digitized"])))

    # Load the stakes geojson
    stakes = gpd.GeoDataFrame.from_features(json.loads("\n".join(data["stakes_digitized"])))
    # Make sure the columns have the right names (it was a bit inconsistent..)
    stakes.rename(columns={"height": "stake_height_cm", "snow_depth": "snow_depth_cm"}, inplace=True)

    # Read the densities csv
    densities = pd.read_csv(io.StringIO("\n".join(data["densities_digitized"])))
    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["from"], densities["to"])

    # Parse all dates as datetimes
    for dataframe in [densities, probings, stakes]:
        dataframe["date"] = pd.to_datetime(dataframe["date"])

    # Initialize the geometry column with nans
    densities["geometry"] = pd.NA
    # For all measurements, see if the note contains the keywords "by stake XX",
    # whereby that stake's coordinate will be taken.
    for i, row in densities.iterrows():
        if pd.isna(row["note"]) or "by stake " not in row["note"].lower():
            continue
        nearby_stake = row["note"][row["note"].lower().index("by stake") + 8 :].strip()
        if " " in nearby_stake:
            nearby_stake = nearby_stake[: nearby_stake.index(" ")]
        if "." in nearby_stake:
            nearby_stake = nearby_stake[: nearby_stake.index(".")]

        try:
            densities.loc[i, "geometry"] = stakes.loc[stakes["stake_id"] == nearby_stake, "geometry"].iloc[0]
        except IndexError as exception:
            raise ValueError(f"Could not find stake {nearby_stake} in {stakes['stake_id'].unique()}") from exception

    # Note the parser version for debugging purposes
    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "1998_version"
    return probings, stakes, densities


def parse_1999_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the "1999" data format.

    The 1999 data format consists of three excel documents. It is similar to later formats, but not identical.
    A quirk of the data is that all coordinates are local. This is handled by the 'parse_coordinates' function.
    Also, neither the summer nor the winter measurement date is known, so this is guessed.

    The version year has nothing to do with when it was created;
    it is the first year it was encountered going back in time.
    For example, 1997 may use the 1998 parser, but 1999 will not use the 1998 parser.

    :param data: The data to parse.
    :param year: Which year the data is associated with.

    :returns: The probings, stakes and densities of that year.
    """
    # There are two density tables beside each other, so these should be split.
    raw_density_dfs = [data["Graf"].iloc[:, :2], data["Graf"].iloc[:, 3:5]]
    density_dfs: list[pd.DataFrame] = []
    for i, density_df in enumerate(raw_density_dfs):
        density_df = set_header(density_df, ["Djup", "Densitet"])
        parsed_density_df = pd.DataFrame(columns=["depth_cm", "density"])
        for j, row in density_df.iterrows():
            if j % 2 != 0:
                continue
            parsed_density_df.loc[j] = pd.Interval(density_df.loc[j - 1, "Djup"], row["Djup"]), row["Densitet"]

        parsed_density_df["location"] = f"hole_{i}"

        density_dfs.append(parsed_density_df)

    densities = pd.concat(density_dfs, ignore_index=True)
    densities["date"] = standard_winter_date(year)

    # Parse the probing measurements.
    probings = set_header(data["AB"], ["cm w eq"]).iloc[:, :3]
    probings.columns = ["X", "Y", "snow_depth_cm"]
    probings = probings[is_number(probings["snow_depth_cm"])]
    probings["geometry"] = parse_coordinates(probings["X"], probings["Y"])
    probings["date"] = standard_winter_date(year)

    # The stakes are in the approximate format: "ID, X, Y, winter values, summer values"
    stakes_raw = set_header(data["Stakar vår"], ["X", "Y", "Djup"])
    stakes_raw = stakes_raw[is_number(stakes_raw["Höjd"])]
    stakes_raw["geometry"] = parse_coordinates(stakes_raw["X"], stakes_raw["Y"])
    stakes_raw["stake_id"] = "M" + stakes_raw.index.astype(str)

    # Split the raw data into winter and summer measurements.
    stakes_w = stakes_raw.rename(columns={"Höjd": "stake_height_cm", "Djup": "snow_depth_cm"})[
        ["stake_id", "stake_height_cm", "snow_depth_cm", "geometry"]
    ]
    stakes_w["date"] = standard_winter_date(year)

    stakes_s = stakes_raw.rename(columns={"Höjd höst": "stake_height_cm"})[["stake_id", "stake_height_cm", "geometry"]]
    stakes_s["snow_depth_cm"] = 0
    stakes_s["date"] = standard_summer_date(year)

    # Concatenate the split winter and summer measurements.
    stakes = pd.concat([stakes_w, stakes_s], ignore_index=True)
    # Note the parser version for debugging purposes
    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "1999_version"
        if "note" not in dataframe:
            dataframe["note"] = ""
        dataframe["note"] += " Summer and winter date is guessed"
    return probings, stakes, densities


def parse_2001_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the "2001" data format.

    The 2001 data format consists of three excel documents.
    The winter and summer dates are not recorded, so these are guessed.

    The version year has nothing to do with when it was created;
    it is the first year it was encountered going back in time.
    For example, 1997 may use the 1998 parser, but 1999 will not use the 1998 parser.

    :param data: The data to parse.
    :param year: Which year the data is associated with.

    :returns: The probings, stakes and densities of that year.
    """
    # Read the snow probings
    probings = set_header(data["Sond_koord"], ["X", "Y"])
    probings.rename(columns={"Sondvärde": "snow_depth_cm"}, inplace=True)
    probings["geometry"] = parse_coordinates(probings["X"], probings["Y"])
    probings["date"] = standard_winter_date(year)
    probings["note"] = "The winter date is guessed"

    # The stakes are in the approximate format: "ID, X, Y, winter values, summer values"
    stakes_raw = set_header(data["stake_koord"], ["X", "Y"])
    cols = [str(col).strip() for col in stakes_raw.columns]
    # The column after the "Vinter" column is unnamed, but contains the winter snow depths.
    cols[cols.index("Vinter") + 1] = "snow_depth_cm"

    # Loop over all columns and find the one with at least three dashes ("-") in it. The stake_id has a dash in it,
    # and there are at least three stakes in the data
    for i in range(stakes_raw.shape[1]):
        if len(str(stakes_raw.iloc[:, i]).split("-")) >= 3:
            cols[i] = "stake_id"
            break
    stakes_raw.columns = cols

    stakes_raw = stakes_raw[is_number(stakes_raw["Vinter"])]
    stakes_raw["geometry"] = parse_coordinates(stakes_raw["X"], stakes_raw["Y"])

    # Split the winter and summer values
    stakes_w = stakes_raw.rename(columns={"Vinter": "stake_height_cm"})
    stakes_w["date"] = standard_winter_date(year)

    stakes_s = stakes_raw.rename(columns={"Sommar": "stake_height_cm"})
    stakes_s["date"] = standard_summer_date(year)
    stakes_s["snow_depth_cm"] = 0

    stake_cols = ["stake_id", "stake_height_cm", "snow_depth_cm", "date", "geometry"]
    stakes = pd.concat([stakes_w[stake_cols], stakes_s[stake_cols]], ignore_index=True)

    stakes["note"] = "The winter and summer dates are guessed"

    densities = set_header(data["Blad1"], ["Djup", "Vikt"])
    densities = densities[is_number(densities["Djup"])]

    # The length of the sample is not noted if it's standard (25 cm)
    densities["Längd"] = densities["Längd"].fillna(25)

    densities["depth_cm"] = pd.IntervalIndex.from_arrays(
        left=densities["Djup"] - densities["Längd"], right=densities["Djup"]
    )

    densities = densities.rename(columns={"Densitet": "density"})[["depth_cm", "density"]]
    densities["date"] = standard_winter_date(year)
    densities["note"] = "The winter date is guessed"

    # Note the parser version for debugging purposes
    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2001_version"
    return probings, stakes, densities


def parse_2002_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the "2002" data format.

    The 2002 data format consists of three excel documents.

    The version year has nothing to do with when it was created;
    it is the first year it was encountered going back in time.
    For example, 1997 may use the 1998 parser, but 1999 will not use the 1998 parser.

    :param data: The data to parse.
    :param year: Which year the data is associated with.

    :returns: The probings, stakes and densities of that year.
    """
    # Read the snow probings
    probings = set_header(data["marma2002"], ["Date", "Sounding"])
    probings.columns = [str(col).strip() for col in probings.columns]
    # The snow depth is recorded in m; convert this to cm
    probings["snow_depth_cm"] = probings["Sounding"] * 100
    probings["geometry"] = parse_coordinates(probings["Easting"], probings["Northing"])
    # Some notes exist that may or may not be interesting.
    probings["note"] = (
        "Prober: " + probings["Sounder"] + ". Alternative: " + (probings["Alternative"] * 100).astype(str)
    )
    # Convert the probings to a GeoDataFrame and only keep the interesting columns.
    probings = gpd.GeoDataFrame(
        probings.rename(columns={"Date": "date"})[["snow_depth_cm", "date", "note", "geometry"]], crs=3006
    )
    # Some probings lack a date for some reason, so the closest date is assigned.
    probings["date"] = probings["date"].ffill()

    # The stakes are in the approximate format: "ID, Easting, Northing, winter values, summer values"
    stakes_raw = set_header(data["Blad1"], ["Date", "Height", "Sounder"])
    cols = [str(col).strip() for col in stakes_raw.columns]
    # The unnamed column after the VT-02 column is the spring stake height
    cols[list(data["Blad1"].iloc[0].values).index("VT-02") + 1] = "spring_height"
    cols[list(data["Blad1"].iloc[0].values).index("HT-02")] = "fall_height"
    stakes_raw.columns = cols

    stakes_raw["note"] = "Prober: " + stakes_raw["Sounder"].astype(str)
    stakes_raw["geometry"] = parse_coordinates(stakes_raw["Easting"], stakes_raw["Northing"])

    # Create the stakes output with only spring measurements first.
    stakes = stakes_raw.rename(
        columns={"Date": "date", "WPT": "stake_id", "spring_height": "stake_height_cm", "d": "snow_depth_cm"}
    )[["stake_id", "date", "stake_height_cm", "snow_depth_cm", "note", "geometry"]]

    # Then get the summer measurements.
    stakes_s = stakes_raw.rename(columns={"WPT": "stake_id", "fall_height": "stake_height_cm"})[
        ["stake_id", "stake_height_cm", "geometry"]
    ]
    stakes_s[["date", "note", "snow_depth_cm"]] = standard_summer_date(year), pd.NA, 0
    stakes_s["note"] += " Summer date is guessed."

    # Finally, append the summer measurements to the output
    stakes = stakes.append(stakes_s, ignore_index=True)
    stakes = stakes[~stakes["stake_id"].isna()]

    # There are multiple density pits in the same sheet, so these need to be split individually.
    density_df_breaks = pd.DataFrame(columns=["start"])
    # They also contain coordinates for once!
    density_coordinates = pd.DataFrame(columns=["easting", "northing"])
    density_i = 0
    # Loop over each row, extract the coordinates, and mark the breaks between the density pits.
    for i, row in data["Densitet"].iterrows():
        row_str = str(row)
        if "Koordinater" in row_str:
            density_coordinates.loc[density_i] = row[[1, 2]].values
            continue
        if not all(s in row_str for s in ["Densitet", "Cylinder"]):
            continue

        density_df_breaks.loc[density_i] = i
        density_i += 1
    density_df_breaks["end"] = density_df_breaks["start"].shift(-1).fillna(data["Densitet"].shape[0])

    density_coordinates = parse_coordinates(
        density_coordinates["easting"].values, density_coordinates["northing"].values
    )
    density_dfs: list[pd.DataFrame] = []
    # Loop over the row breaks, extract the data from the density sheet, and parse it.
    for i, row in density_df_breaks.iterrows():
        density_df = set_header(data["Densitet"].iloc[row["start"] : row["end"]], ["Djup (cm)"])
        density_df = density_df[~density_df["Densitet (g/cm3)"].isna()]

        # Of course, there are two different ways that depth is expressed...
        if "Bottendjup (cm)" in density_df:
            density_df["depth_cm"] = pd.IntervalIndex.from_arrays(
                left=density_df["Topp Djup (cm)"], right=density_df["Bottendjup (cm)"]
            )
        else:
            density_df["depth_cm"] = pd.IntervalIndex.from_arrays(
                left=density_df["Djup (cm)"] - density_df["Längd snödel (cm)"], right=density_df["Djup (cm)"]
            )

        density_df["date"] = probings["date"].max()
        density_df["geometry"] = density_coordinates[i]
        density_dfs.append(
            density_df.rename(columns={"Densitet (g/cm3)": "density"})[["depth_cm", "density", "date", "geometry"]]
        )
    # Concatenate the different density pit measurements.
    densities = pd.concat(density_dfs, ignore_index=True)

    # Note the parser version for debugging purposes
    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2002_version"

    return probings, stakes, densities


def parse_2003_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the "2003" data format.

    This interesting data format has two excel documents a separate coordinate file!
    To make things exciting, the coordinates' decimal point is also in the wrong place by three digits!
    The winter stake measurements are expressed as comments during the snow probings.
    The summer stake measurements are missing, but the calculated ablation values exist.
    Therefore, a reverse-engineering of the ablation values and winter stakes is done to get the summer heights.

    The version year has nothing to do with when it was created;
    it is the first year it was encountered going back in time.
    For example, 1997 may use the 1998 parser, but 1999 will not use the 1998 parser.

    :param data: The data to parse.
    :param year: Which year the data is associated with.

    :returns: The probings, stakes and densities of that year.
    """
    # Read the probing data. The "wpt" will be correlated with the coordinate file to get coordinates.
    probings = data["sondering"]
    date_str = probings.iloc[0, 1]
    probings = set_header(probings, ["wpt", "d"])
    probings["wpt"] = probings["wpt"].astype(int)

    gps_coords = gpd.GeoDataFrame()
    for row in data["marma03wpt"]:
        # The columns with data can be split by 5 "multiple consecutive whitespace" delimiters.
        try:
            wpt_name, date_str, time_str, easting, northing = re.split(r"[\s]+", row.replace("\t", " "))[:5]
        # If it fails with "not enough values to unpack", it's a header or something else.
        except ValueError as exception:
            if "not enough values to unpack" not in str(exception):
                raise exception
            continue

        # This is to remove headers or other lines that happened to have 5 whitespaces
        if not is_number(wpt_name):
            continue

        # Non-ISO-8601 dates should be illegal
        date = datetime.datetime.strptime(date_str + " " + time_str, "%d-%b-%y %H:%M")
        # The coordinates are off by a factor of 1000, obviously!
        gps_coords.loc[gps_coords.shape[0], ["wpt", "date", "easting", "northing"]] = (
            wpt_name,
            date,
            float(easting) * 1e3,
            float(northing) * 1e3,
        )

    gps_coords["geometry"] = parse_coordinates(gps_coords["easting"], gps_coords["northing"])
    gps_coords["wpt"] = gps_coords["wpt"].astype(int)

    # Get the GPS coordinates by joining on the wpt column, and rename some columns
    probings = probings.merge(gps_coords, on="wpt").rename(
        columns={"d": "snow_depth_cm", "kommentar (för orientering, mest)": "note"}
    )

    # Remove invalid values and keep only the interesting columns
    probings = probings[is_number(probings["snow_depth_cm"])][["snow_depth_cm", "note", "date", "geometry"]]

    # Some dates are most likely wrong; they are from 2002 supposedly, which doesn't make sense as another...
    # ... document says these are from April 2003!
    # Therefore, they are removed and filled by the closest 2003-value
    probings.loc[probings["date"].apply(lambda d: d.year == 2002), "date"] = pd.NA
    probings["date"] = probings["date"].bfill()

    stakes = gpd.GeoDataFrame(
        columns=["stake_id", "snow_depth_cm", "stake_height_cm", "date", "note", "geometry"], crs=3006
    )
    # The winter stake measurements are comments in the probings file...
    # Loop through all comments and parse the notes to get the stakes
    # Some stakes are not given IDs, so these will be given an "unknown" number
    unknowns_n = 0
    for _, probing in probings.iterrows():
        # The stake heights are either colon or equal-sign separated, because who needs consistency?
        if not any(s in str(probing["note"]) for s in ["h=", "h:"]):
            continue
        stake_height_cm = np.nan
        stake_id = ""

        for word in str(probing["note"]).split(" "):
            # The stake_id is always 4 characters long and contains a dash
            if len(word) == 4 and "-" in word:
                stake_id = word
            elif "h:" in word or "h=" in word:
                stake_height_cm = int(word.replace("h:", "").replace("h=", "").replace(",", ""))
                # Ugly hardcoding of note ("300 was removed")
                if "300 togs av" in probing["note"]:
                    stake_height_cm -= 300
        # If no stake_id was found in the note, add it as one of the unknowns.
        if len(stake_id) == 0:
            stake_id = f"unknown_{unknowns_n}"
            unknowns_n += 1

        # Take the date, probing depth, coordinate, and note from the
        stakes.loc[stakes.shape[0]] = (
            stake_id,
            probing["snow_depth_cm"],
            stake_height_cm,
            probing["date"],
            probing["note"],
            probing["geometry"],
        )

    # Extract all of the stake names (sorted by the Easting coordinate)
    stakes["x"] = stakes["geometry"].apply(lambda geom: geom.x)
    sorted_stake_names = [name for name in stakes.sort_values("x")["stake_id"].values if "unknown" not in name]
    stakes.drop(columns="x", inplace=True)
    # There are no stake measurements anywhere from the summer, only the calculated ablation...
    summer_ablation = set_header(data["stake_koord"], ["X", "Y", "Ablation"]).sort_values("X")
    # We assume that these are the same as in the "stake_koord" sheet
    assert len(sorted_stake_names) == summer_ablation.shape[0]
    summer_ablation["stake_id"] = sorted_stake_names

    # To derive the summer heights, first the densities have to be parsed.
    density_dfs = []
    for key in data:
        if " dens" not in key:
            continue
        location_row = " ".join(
            map(str, data[key].loc[data[key].apply(lambda row: "densitet" in str(row), axis=1)].iloc[0].values)
        )
        location = ""
        for word in location_row.split(" "):
            if len(word) == 4 and "-" in word:
                location = word
                break

        density_df = set_header(data[key], ["run", "densitet"])
        density_df["note"] = density_df.iloc[:, list(density_df.columns).index("densitet") + 1 :].apply(
            lambda row: " ".join(map(str, np.where(pd.isna(row.values), "", row.values.astype(str))))
            .replace("\t", "")
            .strip(),
            axis=1,
        )
        if "ack.djup" in density_df.columns:
            density_df["depth_cm"] = pd.IntervalIndex.from_breaks(np.r_[[0], density_df["ack.djup"].values])
        else:
            density_df["depth_cm"] = pd.IntervalIndex.from_breaks(np.r_[[0], density_df["längd"].cumsum()])

        density_df["date"] = stakes["date"].iloc[-1]
        density_df.rename(columns={"densitet": "density"}, inplace=True)
        density_df["location"] = location
        density_df = density_df[~density_df["density"].isna()]

        density_dfs.append(density_df)

    density_cols = ["location", "depth_cm", "density", "date", "note"]
    densities = pd.DataFrame(
        np.concatenate([density_df[density_cols].values for density_df in density_dfs]), columns=density_cols
    )

    # The mean densities (average weighted by sample length) are used to back-calculate the ablation value.
    mean_densities = densities.groupby(densities["location"]).apply(
        lambda df: np.average(
            df["density"].astype(float),
            weights=[i.length for i in df["depth_cm"].values],
        )
    )
    # I couldn't find a way to make this work better, so here's an ugly hardcoded mapping to the stakes
    summer_ablation["snow_density"] = [mean_densities.iloc[0], mean_densities.iloc[-1], mean_densities.iloc[-1]]
    summer_ablation[["winter_snow_depth", "winter_height", "geometry"]] = stakes.loc[
        stakes["stake_id"].isin(summer_ablation["stake_id"]), ["snow_depth_cm", "stake_height_cm", "geometry"]
    ].values
    # The winter snow w.e. is calculated by the snow depth times its mean density
    summer_ablation["winter_snow_we"] = summer_ablation["winter_snow_depth"] * summer_ablation["snow_density"]
    # Here, a major assumption is made: during summer, the snow densifies to 0.6 m w.e. / m of snow. If the ablation (in m w.e.) is lower than the snow height in w.e., there is densified snow left in the summer.
    # If the ablation is higher, all snow is gone and the value should be 0
    # Note that this only seems to be the case at the highest stake, so the major assumption will only be valid there.
    summer_ablation["summer_snow_w_abl"] = np.clip(
        (summer_ablation["winter_snow_we"] - summer_ablation["Ablation"]) / 0.6, 0, 100000
    )
    # Another assumption is made here: ice has a density of 0.9 m w.e. / m. This is the winter snow depth minus the ...
    # ... total ablation, and converted using the density assumption.
    summer_ablation["ice_melt"] = np.clip(
        (summer_ablation["winter_snow_we"] - summer_ablation["Ablation"]) / 0.9, -100000, 0
    )

    # The stake height over ice in the winter is the winter snow depth plus the winter height above the snow
    summer_ablation["winter_height_above_ice"] = summer_ablation["winter_snow_depth"] + summer_ablation["winter_height"]
    # The summer height is the winter height minus the snow and ice gain/loss
    summer_ablation["summer_height"] = summer_ablation["winter_height_above_ice"] - (
        summer_ablation["summer_snow_w_abl"] + summer_ablation["ice_melt"]
    )

    summer_ablation["date"] = standard_summer_date(year)
    summer_ablation[
        "note"
    ] = "Only w.e. ablation values were available so the stake height is derived from this and the winter values. A compacted snow density of 0.6 was assumed, and 0.9 for pure ice. The summer date is guessed."

    stakes = stakes.append(
        summer_ablation.rename(columns={"summer_snow_w_abl": "snow_depth_cm", "summer_height": "stake_height_cm"})[
            stakes.columns
        ]
    )
    # Note the parser version for debugging purposes
    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2003_version"

    return probings, stakes, densities


def parse_2004_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the "2004" data format.

    The data format consists of two excel documents.
    The winter and summer dates are missing.

    The version year has nothing to do with when it was created;
    it is the first year it was encountered going back in time.
    For example, 1997 may use the 1998 parser, but 1999 will not use the 1998 parser.

    :param data: The data to parse.
    :param year: Which year the data is associated with.

    :returns: The probings, stakes and densities of that year.
    """
    probings = set_header(data["gpscombi"], ["Stake", "Tid/Datum", "Easting"]).rename(
        columns={"Tid/Datum": "date", "Stake": "stake_id", "Kom.": "note"}
    )
    # For some reason, the coordinates are off by a factor of 1000
    probings[["Easting", "Northing"]] *= 1e3
    probings["geometry"] = parse_coordinates(probings["Easting"], probings["Northing"])

    probings["snow_depth_cm"] = probings["Snödjup (m)"] * 100
    probings = probings.loc[is_number(probings["snow_depth_cm"]), ["snow_depth_cm", "note", "date", "geometry"]]

    # At least one date was wrong and therefore not parsed as datetime, so just replace those with the last date
    probings.loc[probings["date"].apply(lambda s: isinstance(s, str)), "date"] = probings["date"].iloc[-1].date()

    stakes = set_header(data["Rådata"], ["Stake", "East", "North"]).rename(columns={"Stake": "stake_id"}).iloc[:, :7]
    stakes["geometry"] = parse_coordinates(stakes["East"], stakes["North"])

    stakes_w = stakes.rename(columns={"Snödjup": "snow_depth_cm", "vinterhöjd": "stake_height_cm"})
    stakes_w["date"] = standard_winter_date(year)
    stakes_w["note"] = "The winter date is guessed."

    stakes_s = stakes.rename(columns={"sommarhöjd": "stake_height_cm"})
    stakes_s["snow_depth_cm"] = 0
    stakes_s["date"] = standard_summer_date(year)
    stakes_s["note"] = "The summer date is guessed"

    stakes = pd.concat([stakes_w, stakes_s])[
        ["stake_id", "snow_depth_cm", "stake_height_cm", "date", "note", "geometry"]
    ]

    densities = set_header(data["Densitet"], ["Snödjup", "Densitet (kg/m3)"]).rename(
        columns={"Densitet (kg/m3)": "density"}
    )
    densities = densities[is_number(densities["density"])]
    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities.iloc[:, 1], densities["Snödjup"])
    densities = densities[~densities["depth_cm"].isna()]
    densities["date"] = probings["date"].iloc[0].date()
    densities["note"] = "The winter date is guessed."

    densities = densities[["depth_cm", "density", "date", "note"]]

    # Note the parser version for debugging purposes
    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2004_version"

    return probings, stakes, densities


def parse_2005_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the "2005" data format.

    The version year has nothing to do with when it was created;
    it is the first year it was encountered going back in time.
    For example, 1997 may use the 1998 parser, but 1999 will not use the 1998 parser.

    :param data: The data to parse.
    :param year: Which year the data is associated with.

    :returns: The probings, stakes and densities of that year.
    """
    probings = set_header(data["sond"], ["Alt", "Snödjup (m)"])
    cols = [str(col).strip() for col in probings.columns]
    cols[-1] = "note"
    probings.columns = cols

    probings.loc[~probings["Alt"].isna(), "Snödjup (m)"] *= 100
    probings["geometry"] = parse_coordinates(probings["Easting"], probings["Northing"])

    probings = probings[["Snödjup (m)", "note", "geometry"]]
    probings.columns = ["snow_depth_cm", "note", "geometry"]
    probings["date"] = standard_winter_date(year)
    probings["note"] = "The winter date is guessed"

    densities = set_header(data["Density"], ["From", "To", "Density"])
    densities = densities[is_number(densities["Density"])]
    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["From"], densities["To"])
    densities = densities[["depth_cm", "Density"]].rename(columns={"Density": "density"})
    densities["date"] = standard_winter_date(year)
    densities["note"] = "The winter date is guessed"

    stakes = set_header(data["Stakar"], ["Stake", "Easting"])

    stakes["geometry"] = parse_coordinates(stakes["Easting"], stakes["Norting"])
    i = 0
    newcols = []
    for col in stakes.columns:
        if col == "H":
            newcols.append(f"H{i}")

            i += 1
        else:
            newcols.append(col)

    stakes.columns = newcols

    stakes_w = stakes[["Stake", "H0", "D", "geometry"]].rename(
        columns={"Stake": "stake_id", "H0": "stake_height_cm", "D": "snow_depth_cm"}
    )
    stakes_w["date"] = standard_winter_date(year)
    stakes_w["note"] = "The winter date is guessed"

    stakes_s = stakes[["Stake", "H1", "geometry"]].rename(columns={"Stake": "stake_id", "H1": "stake_height_cm"})
    stakes_s["snow_depth_cm"] = 0.0
    stakes_s["date"] = standard_summer_date(year)
    stakes_s["note"] = "The summer date is guessed"

    stakes = pd.concat([stakes_w, stakes_s], ignore_index=True)

    # Note the parser version for debugging purposes
    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2005_version"
    return probings, stakes, densities


def parse_2011_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the "2013" data format.

    The 2011 data format is exactly the same as in 2013, but the late-summer stake readings are missing.
    The ablation values exist, however, so these can be used to back-track the stake readings.

    :param data: The data to parse.
    :param year: Which year the data is associated with.

    :returns: The probings, stakes and densities of that year.
    """
    probings, stakes, densities = parse_2013_version(data, year=year)

    # Sort by date so the winter reading can be extracted easily (the first matching value)
    stakes.sort_values("date", inplace=True)
    # Initialize the note column
    stakes["note"] = pd.NA

    # This is the ablation per stake in meters
    ablation = pd.read_csv(io.StringIO("\n".join(data["ablation"])), index_col=0, squeeze=True)

    # The mean (average weighted by sample length) density will be used to get the w.e. of the snow
    mean_density = np.average(
        densities["density"].astype(float), weights=densities["depth_cm"].apply(lambda i: i.length)
    )

    for stake_id, abl in ablation.iteritems():
        winter_reading = stakes.loc[stakes["stake_id"] == stake_id].iloc[0].copy()
        winter_reading["winter_snow_we"] = winter_reading["snow_depth_cm"] * mean_density

        # Based on experimentation, no snow was retained, so this doesn't have to be as convoluted as 2003
        if winter_reading["winter_snow_we"] > abl * 100:
            raise NotImplementedError("An assumption was made that all snow was lost")

        # The summer height is the ablation (conv. to cm) minus snow in w.e., then the density is applied
        summer_height = (abl * 100 - winter_reading["winter_snow_we"]) / 0.9

        # Find the summer value in the stakes dataframe and assign the calculated value
        stakes.loc[
            (stakes["stake_id"] == stake_id) & (stakes["date"].apply(lambda d: d.month > 6)),
            ["stake_height_cm", "note"],
        ] = (
            summer_height,
            (
                "The stake reading is missing, but the calculated ablation value exists."
                "The reading was back-calculated using the mean snow density and an ice density of 0.9"
            ),
        )

    stakes["parser"] = "2011_version"
    return probings, stakes, densities


def parse_2013_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the "2013" data format.

    The 2013 data format consists of one excel document.
    The summer date for 2013 is missing.

    The version year has nothing to do with when it was created;
    it is the first year it was encountered going back in time.
    For example, 1997 may use the 1998 parser, but 1999 will not use the 1998 parser.

    :param data: The data to parse.
    :param year: Which year the data is associated with.

    :returns: The probings, stakes and densities of that year.
    """
    # Read the densities
    densities = set_header(data["Density"], ["From", "To", "Density"])
    winter_date = densities.iloc[0, -1].date()
    densities = densities[densities["Density"].apply(lambda s: is_number(s) and float(s) != 0.0)]
    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["From"], densities["To"])
    densities = densities[["depth_cm", "Density"]].rename(columns={"Density": "density"})
    densities["date"] = winter_date

    probings = set_header(data["Probing data"], ["Easting", "Northing"])
    probings = probings[is_number(probings["Snow depth"])]
    probings["geometry"] = parse_coordinates(probings["Easting"], probings["Northing"])

    probings = probings[["Snow depth", "Note", "geometry"]]
    probings.columns = ["snow_depth_cm", "note", "geometry"]
    probings["date"] = winter_date

    stakes = set_header(data["Summer balance calc"], ["Stake", "Easting", "Northing", "Elevation"])
    stakes = stakes[is_number(stakes["HW"])]
    stakes["geometry"] = parse_coordinates(stakes["Easting"], stakes["Northing"])

    stakes_w = stakes[["Stake", "HW", "DW", "geometry"]].rename(
        columns={"Stake": "stake_id", "HW": "stake_height_cm", "DW": "snow_depth_cm"}
    )
    stakes_w["surface"] = "S"
    stakes_w["date"] = winter_date

    stakes_s = stakes[["Stake", "HL", "Surface", "geometry"]].rename(
        columns={"Stake": "stake_id", "HL": "stake_height_cm", "Surface": "surface"}
    )
    stakes_s["snow_depth_cm"] = 0.0
    stakes_s["date"] = standard_summer_date(year)

    stakes = pd.concat([stakes_w, stakes_s])

    # Convert from m to cm
    stakes[["stake_height_cm", "snow_depth_cm"]] *= 100

    # Note the parser version for debugging purposes
    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2013_version"
    return probings, stakes, densities


def parse_2016_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the "2016" data format.

    The 2016 data format consists of one excel file.

    The version year has nothing to do with when it was created;
    it is the first year it was encountered going back in time.
    For example, 1997 may use the 1998 parser, but 1999 will not use the 1998 parser.

    :param data: The data to parse.
    :param year: Which year the data is associated with.

    :returns: The probings, stakes and densities of that year.
    """
    # Densities are made first (in 2016, only the densities have a winter date)
    # In 2008, there are two sheets, whereby the proper density data are in Density_1
    # In 2010, the densities are in a separate csv.
    if int(year) == 2010:
        # Rename the columns to fit the other sheets' names.
        densities = pd.read_csv(io.StringIO("\n".join(data["densities"]))).rename(
            columns={"from": "From", "to": "To", "density": "Density"}
        )
    else:
        # Therefore, loop through both and identify where data exist.
        for key in ["Density", "Density_1"]:
            if key not in data:
                continue
            densities = data[key]
            densities = set_header(densities, ["From", "To", "Density"])
            densities = densities[is_number(densities["Density"])]
            if densities["Density"].isna().all():
                continue
            break

        else:
            raise ValueError("Could not find valid density values.")

    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["From"], densities["To"])
    densities = densities[["depth_cm", "Density"]].rename(columns={"Density": "density"})

    probings = set_header(data["Probing data"], column_values=["Easting", "Northing"])
    probings.rename(columns={"#": "stake_id", "Snow depth": "snow_depth_cm", "Note": "note"}, inplace=True)
    probings = probings[is_number(probings["snow_depth_cm"])]

    # If there are two easting columns, there are coordinates in both RT90 and SWEREF99TM
    if len(list(filter(lambda col: col == "Easting", probings.columns))) > 1:
        probings["geometry"] = parse_coordinates(probings.iloc[:, 3], probings.iloc[:, 4])
    else:
        probings["geometry"] = parse_coordinates(probings["Easting"], probings["Northing"])
    probings = probings[["stake_id", "snow_depth_cm", "note", "geometry"]]

    # The date can be a bit hard to find as it is defined in different places over different years (2016, 2015...)
    try:
        # This may work in some cases. If not, and IndexError is reached an another more convoluted search is made.
        winter_date = densities.iloc[:, -1][densities.iloc[:, -1].apply(lambda d: hasattr(d, "year"))].iloc[0].date()
    except IndexError:
        # Convert the data into a huge lower-case string.
        data_str = data["Probing data"].astype(str).applymap(lambda s: s.lower())

        # All cells that contain "apr" (for April) or all datetime objects are possible dates.
        potential_dates = data_str.values[
            data_str.applymap(lambda s: "apr" in s).values
            | data["Probing data"].applymap(lambda d: isinstance(d, datetime.datetime))
        ]

        # There could also be dates in the density sheet!
        potential_dates = np.append(
            potential_dates,
            data["Density"].values[data["Density"].applymap(lambda d: isinstance(d, datetime.datetime))],
        )
        # Initialize the winter date variable, then try to convert the found dates until one works.
        winter_date = None
        for potential_date in potential_dates:
            try:
                winter_date = pd.Timestamp(potential_date).date()
            except ValueError:
                continue
            break

        # If no potential date was convertible, raise an error
        if winter_date is None:
            raise ValueError("Could not find winter date")

    # At this point, the winter date exists (or an error had been raised)
    probings["date"] = winter_date
    densities["date"] = winter_date

    stakes_raw = data["Stake ablation"]
    stakes = pd.DataFrame()

    # Here, some slight repetition of the 'set_header()' functionality is made. This is because some intermediate
    # information is still needed.
    mask = np.sum([stakes_raw == value for value in ["Stake", "Easting", "Northing", "HW"]], axis=0).sum(axis=1) >= 4

    col_index = np.argwhere(mask).ravel()[-1]
    # Set the column row to the columns
    stakes_raw.columns = stakes_raw.loc[col_index]
    stakes_raw = stakes_raw.dropna(how="all")

    # Extract the winter readings first
    stakes = (
        stakes_raw[["Stake", "Easting", "Northing", "HW", "DW"]]
        .copy()
        .rename(columns={"HW": "height", "DW": "winter_snow_depth"})
    )
    stakes["surface"] = "S"
    # Only keep the data after the column row.
    stakes = stakes.loc[:, ~stakes.columns.duplicated()]
    stakes = stakes[stakes["Stake"] != "#"]
    stakes = stakes.loc[col_index + 1 :]
    stakes["date"] = winter_date

    # The summer readings are in columns left-to-right. The date is in one of the columns' headers.
    # Sometimes, the date is wrong so it has to be validated.
    previous_date = pd.Timestamp(year=1900, month=1, day=1)
    for i, col in enumerate(stakes_raw):
        if "H" not in str(col):
            continue
        if not col.replace("H", "").isnumeric() and col != "HS":
            continue
        subset = stakes_raw.iloc[:, [i, i + 1]].copy()
        subset["Stake"] = stakes_raw.iloc[:, list(stakes_raw.columns).index("Stake")]

        # The datetimes are all over the place (and sometimes incorrect), so manual parsing is better
        date_str = str(subset.iloc[2, 1])
        if date_str == "nan":
            if "HS" in subset:
                summer_date = standard_summer_date(year)
            else:
                continue
        else:
            month = date_str[5:7]
            day = date_str[8:10]
            summer_date = pd.Timestamp(year=year, month=int(month), day=int(day))

        # TODO: Explain what's going on here. I'm lost!
        if summer_date < previous_date and col != "HS":
            summer_date = pd.Timestamp(year=summer_date.year + 1, month=summer_date.month, day=summer_date.day)

        previous_date = summer_date

        for j, row in subset.iterrows():
            # Iteratively find the header row by identifying the units
            if all((any(row.values == "(m)"), any(row.values == "(S/SI/I/F)"))):
                data_header_row = j
                break
        subset = subset.iloc[data_header_row + 1 :]
        # Remove rows where no stake height exists
        subset = subset[~subset.iloc[:, 0].isna()]
        subset.columns = ["height", "surface", "Stake"]
        subset["date"] = summer_date

        stakes = pd.concat([stakes, subset], ignore_index=True)

    stakes = stakes[is_number(stakes["height"])]
    stakes.rename(columns={"Stake": "stake_id"}, inplace=True)
    stakes["stake_height_cm"] = stakes["height"] * 100
    stakes["snow_depth_cm"] = stakes["winter_snow_depth"] * 100

    # Parse the last easting and northing columns (there may be multiple; RT90 and SWEREF99TM)
    stakes["geometry"] = parse_coordinates(
        stakes.loc[:, stakes.columns == "Easting"].iloc[:, -1], stakes.loc[:, stakes.columns == "Northing"].iloc[:, -1]
    )
    # Standardize the way the identification is written (in some: M-1, in others: M1)
    stakes["stake_id"] = stakes["stake_id"].astype(str).str.replace("M-", "M")

    # Note the parser version for debugging purposes
    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2016_version"

    return probings, stakes, densities


def parse_2017_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the "2017" data format.

    The 2017 data format consists of one excel document for many different glaciers, including Mårmaglaciären.

    The version year has nothing to do with when it was created;
    it is the first year it was encountered going back in time.
    For example, 1997 may use the 1998 parser, but 1999 will not use the 1998 parser.

    :param data: The data to parse.
    :param year: Which year the data is associated with.

    :returns: The probings, stakes and densities of that year.
    """
    probings = data["Mårma Sondering"]

    date_str = str(probings.iloc[1, 0])
    date = datetime.date(year, int(date_str[:2]), int(date_str[2:]))

    probings = set_header(probings, ["Date", "Punkt"])
    probings.columns = ["date", "stake_id", "snow_depth_cm", "note"]
    probings["date"] = date
    # It does not contain coordinates, but adding the M allows comparison with 2019 coordinates
    # probings["stake_id"] = "M" + probings["stake_id"]

    stakes = set_header(data["Mårma stakes"], ["Date", "ID", "Snow Depth"])
    columns = {
        "Date": "date",
        "ID": "stake_id",
        "Snow Depth": "snow_depth_cm",
        "Stake Height": "stake_height_cm",
        "Note": "note",
    }
    stakes = stakes.rename(columns=columns)[list(columns.values())].dropna(how="all")

    for dataframe in [stakes, probings]:
        # The naming convention was a little bit different in 2017. Rename e.g. 9C -> M09C
        for i, point in dataframe.iterrows():
            stake_row = ""
            for character in str(point["stake_id"]):
                if character.isnumeric():
                    stake_row += character
                else:
                    break

            stake_col = str(point["stake_id"]).replace(stake_row, "")
            dataframe.loc[i, "stake_id"] = "M" + stake_row.zfill(2) + stake_col

    # Retain only the columns of interest.
    stakes = stakes[list(columns.values())]

    densities = data["Mårma Densitet"]
    date_str = str(densities.iloc[1, 0])
    date = datetime.date(year, int(date_str[:2]), int(date_str[2:]))
    densities = set_header(densities, ["Date", "Från (cm)", "Till (cm)"])
    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["Från (cm)"], densities["Till (cm)"])
    densities = densities[["depth_cm", "Densitet"]].rename(columns={"Densitet": "density"})
    densities["date"] = date

    # Note the parser version for debugging purposes
    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2017_version"

    return probings, stakes, densities


def parse_2019_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the "2019" data format.

    The 2019 data format consists of one excel document.

    The version year has nothing to do with when it was created;
    it is the first year it was encountered going back in time.
    For example, 1997 may use the 1998 parser, but 1999 will not use the 1998 parser.

    :param data: The data to parse.
    :param year: Which year the data is associated with.

    :returns: The probings, stakes and densities of that year.
    """
    probings = data["Mårma_PROBE"]

    winter_date = (
        probings.iloc[0, 1].date() if isinstance(probings.iloc[0, 1], datetime.datetime) else standard_winter_date(year)
    )

    probings = set_header(probings, ["E", "N"])
    probings["geometry"] = parse_coordinates(probings["E"], probings["N"])
    probings = (
        probings.rename(columns={"SNOW DEPTH (cm)": "snow_depth_cm", "PROBE POINT ID": "stake_id", "NOTE": "note"})
        .drop(columns=["#", "E", "N"])
        .dropna(subset=["snow_depth_cm"])
    )
    probings["date"] = winter_date

    stakes = set_header(data["Mårma_STAKES"], ["DATE", "STAKE ID"])
    stakes = stakes[stakes["DATE"].apply(lambda d: hasattr(d, "year") and hasattr(d, "month"))]
    stakes = stakes[list(filter(lambda c: ~pd.isna(c) and str(c).lower() != "nan", stakes.columns))]
    stakes = stakes[stakes.columns[~stakes.columns.duplicated()]].rename(
        columns={
            "STAKE ID": "stake_id",
            "STAKE HEIGHT (cm)": "stake_height_cm",
            "SNOW DEPTH (cm)": "snow_depth_cm",
            "NOTES": "note",
            "SURFACE": "surface",
            "DATE": "date",
        }
    )
    stakes = stakes[["stake_id", "stake_height_cm", "snow_depth_cm", "note", "surface", "date"]]

    densities = data["Mårma_DENSITY"]

    density_date = densities.iloc[0, 1] if isinstance(densities.iloc[0, 1], datetime.datetime) else winter_date
    densities = set_header(densities, ["FROM (cm)", "TO (cm)"])

    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["FROM (cm)"], densities["TO (cm)"])
    densities = densities.rename(columns={"DENSITY": "density"})[["depth_cm", "density"]]
    densities["date"] = density_date

    # Note the parser version for debugging purposes
    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2019_version"

    return probings, stakes, densities


@np.vectorize
def is_number(obj: object) -> bool:
    """
    Validate if an object or array of objects could be represented as numbers.

    NaNs are not considered valid numbers.

    :examples:
    >>> is_number(5)
    True
    >>> is_number("5")
    True
    >>> is_number("five")
    False
    >>> is_number(["5", "five"])
    [True, False]
    >>> is_number(np.nan)
    False
    """
    obj = str(obj)
    try:
        number = float(obj)
        return not np.isnan(number)
    except ValueError:
        return False


def set_header(dataframe: pd.DataFrame, column_values: list[str]) -> pd.DataFrame:
    """
    Identify and set the proper header column in a parsed dataframe.

    For example, the dataframe might look like this:
    >>> df # doctest:+SKIP
        1   2   3   4   5
    0   x   y   z   val note
    1   0.5 0.3 1.  5.  "Test data"

    In this case, the row "0" is the actual header, and should be set appropriately:
    >>> set_header(df, column_values=["x", "y"])  # doctest:+SKIP
        x   y   z   val note
    1   0.5 0.3 1.  5.  "Test data"

    :param dataframe: The dataframe to modify a copy of.
    :param column_values: A list of column names to identify as the header.

    :raises IndexError: If all columns could not be found.

    :returns: A copy of the dataframe with the properly set header.
    """
    new_data = dataframe.copy()

    mask = np.sum([dataframe == value for value in column_values], axis=0).sum(axis=1) >= len(column_values)

    header_index = np.argwhere(mask).ravel()[0]

    new_data.columns = new_data.iloc[header_index, :]
    new_data = new_data.iloc[header_index + 1 :, :]

    return new_data


def parse_coordinates(
    easting: np.ndarray | list[float | int], northing: np.ndarray | list[float | int]
) -> gpd.GeoSeries:
    """
    Parse arrays of coordinates.

    They are assumed to be either SWEREF99TM or RT90 2.5gon V, depending on the size of the easting coord.
    Some cases also have local coordinates which are modified RT90 2.5gon V coordinates.

    :param easting: An array-like of easting coordinates.
    :param northing: An array-like of northing coordinates.

    :returns: A GeoSeries in SWEREF99TM
    """
    easting = np.asarray(easting)
    northing = np.asarray(northing)

    # Generally (at least at Mårmaglaciären), the RT90 easting coordinate is very large (>1e6),
    # ... thus separating it clearly from SWEREF99TM (<1e6)
    # However, if the coordinate is tiny (lower than 10000), it's most likely a local coordinate
    if easting[0] < 1e4:
        # The local coordinates are RT90 2.5gon V coordinates from the bottom left corner of the 1978 map.
        # Their decimal point is also wrong, so they have to be multiplied by 10.
        # How did I figure this out? Trial and error until it worked!
        # It could be that the offsets should have been rounded (1618000, 7555500), but this is the corner-coordinate
        # of the newly georeferenced 1978 map, so this is most likely the true offset.
        crs = 3021

        east_offset, north_offset = (1617990, 7555480)
        scale = 10
    elif easting[0] < 1e6:
        crs = 3006
        east_offset, north_offset = (0, 0)
        scale = 1
    else:
        crs = 3021
        east_offset, north_offset = (0, 0)
        scale = 1

    return gpd.points_from_xy((easting * scale) + east_offset, (northing * scale) + north_offset, crs=crs).to_crs(3006)


def test_parse_coordinates():
    """Test the parse_coordinates() function."""
    # These coordinates should represent approximately the same place
    marma_rt90 = 1621920, 7557435
    marma_sweref = 655350, 7556847
    marma_local = 393, 195.5

    # Convert all to SWEREF99TM
    converted_rt90 = parse_coordinates([marma_rt90[0]], [marma_rt90[1]])[0]
    converted_sweref = parse_coordinates([marma_sweref[0]], [marma_sweref[1]])[0]
    converted_local = parse_coordinates([marma_local[0]], [marma_local[1]])[0]

    # Validate that the SWEREF99TM coordinate is exactly the same after conversion
    assert marma_sweref[0] == converted_sweref.x and marma_sweref[1] == converted_sweref.y

    # Validate that all coordinates represent the same spot (within 1 m)
    assert np.mean(np.abs(np.diff([converted_rt90.x, converted_sweref.x, converted_local.x, marma_sweref[0]]))) < 1
    assert np.mean(np.abs(np.diff([converted_rt90.y, converted_sweref.y, converted_local.y, marma_sweref[1]]))) < 1


def test_is_number():
    """Test the is_number() function."""
    assert is_number("9")
    assert not is_number("jgh")
    assert not is_number(">500")

    assert np.array_equal(is_number(["9", "ladw"]), [True, False])


if __name__ == "__main__":
    probings, stakes, densities = read_all_data(Path("massbalance/"))
    if os.path.isdir("output"):
        shutil.rmtree("output/")
    os.mkdir("output")
    probings.to_file("output/probings.geojson", driver="GeoJSON")
    stakes.to_file("output/stakes.geojson", driver="GeoJSON")
    densities.to_csv("output/densities.csv", index=False)

    with tarfile.open("marma_mb.tar.gz", mode="w:gz") as tar:
        tar.add("output/", arcname="")
        tar.add("dems/", arcname="")
