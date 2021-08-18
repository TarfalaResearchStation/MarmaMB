from __future__ import annotations

import datetime
import os
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_all_data(base_directory: Path):

    year_dirs = filter(lambda d: d.isnumeric(), os.listdir(base_directory))

    probing_dfs = []
    stake_dfs = []
    density_dfs = []
    for year in sorted(year_dirs, reverse=True):

        files = list(filter(lambda s: "xls" in s and "lock" not in s, os.listdir(base_directory.joinpath(year))))
        if len(files) == 0:
            continue

        #if int(year) > 2011:
        #    continue

        with warnings.catch_warnings():

            warnings.filterwarnings("ignore", "Data Validation extension is not supported")
            data: dict[str, pd.DataFrame] = {}
            i = 0
            for filename in files:
                for key, value in pd.read_excel(base_directory.joinpath(year).joinpath(filename), sheet_name=None, header=None).items():
                    key_to_use = key
                    for i in range(1, 100):
                        if key_to_use in data:
                            key_to_use = key + f"_{i}"
                        else:
                            break
                            
                    data[key_to_use] = value

        probings, stakes, densities = figure_out_version(data, int(year))

        stakes = stakes.loc[:, ~stakes.columns.duplicated()]

        probing_dfs.append(probings)
        stake_dfs.append(stakes)
        density_dfs.append(densities)

    densities = pd.concat(density_dfs, ignore_index=True)
    probings = pd.concat(probing_dfs, ignore_index=True)
    stakes = pd.concat(stake_dfs, ignore_index=True)


    print(stakes)


def figure_out_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    try:
        if all(key in data for key in ["Mårma_DENSITY", "Mårma_STAKES", "Mårma_PROBE"]):
            return parse_2019_version(data, year)
        if all(key in data for key in ["Mårma Densitet", "Mårma stakes", "Mårma Sondering"]):
            return parse_2017_version(data, year)
        if all(key in data for key in ["Density", "Stake ablation", "Probing data"]):
            return parse_2016_version(data, year)
        if all(key in data for key in ["Density", "Summer balance calc", "Probing data"]):
            return parse_2013_version(data, year)
        if all(key in data for key in ["Density", "Stakar", "sond"]):
            return parse_2005_version(data, year)
    except Exception as exception:
        print(f"Failed on {year=}")
        raise exception

    raise ValueError(f"Data sheet for {year=} did not match patterns:\n{data}")


def parse_2005_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    probings = set_header(data["sond"], ["Alt", "Snödjup (m)"])
    cols = [str(col).strip() for col in probings.columns]
    cols[-1] = "note"
    probings.columns = cols

    # TODO: Find out the actual date instead of guessing.
    date = datetime.date(year, 4, 15)

    probings.loc[~probings["Alt"].isna(), "Snödjup (m)"] *= 100
    probings["geometry"] = gpd.points_from_xy(probings["Easting"], probings["Northing"], crs=3022).to_crs(3006)

    probings = probings[["Snödjup (m)", "note", "geometry"]]
    probings.columns = ["snow_depth_cm", "note", "geometry"]
    probings["date"] = date

    densities = set_header(data["Density"], ["From", "To", "Density"])
    densities = densities[densities["Density"].apply(lambda s: is_number(s))]
    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["From"], densities["To"])
    densities = densities[["depth_cm", "Density"]].rename(columns={"Density": "density"})
    densities["date"] = date

    stakes = data["Stakar"]

    print(stakes)
    raise NotImplementedError()

def parse_2013_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:


    densities = set_header(data["Density"], ["From", "To", "Density"])
    date = densities.iloc[0, -1].date()
    densities = densities[densities["Density"].apply(lambda s: is_number(s) and float(s) != 0.0)]
    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["From"], densities["To"])
    densities = densities[["depth_cm", "Density"]].rename(columns={"Density": "density"})
    densities["date"] = date

    probings = set_header(data["Probing data"], ["Easting", "Northing"])
    probings = probings[probings["Snow depth"].apply(lambda s: is_number(s))]
    probings["geometry"] = gpd.points_from_xy(probings["Easting"], probings["Northing"], crs=3006)

    probings = probings[["Snow depth", "Note", "geometry"]]
    probings.columns = ["snow_depth_cm", "note", "geometry"]
    probings["date"] = date

    stakes = set_header(data["Summer balance calc"], ["Stake", "Easting", "Northing", "Elevation"])
    stakes = stakes[stakes["HW"].apply(lambda s: is_number(s))]
    stakes["geometry"] = gpd.points_from_xy(stakes["Easting"], stakes["Northing"], crs=3006)

    stakes_w = stakes[["Stake", "HW", "DW", "geometry"]].rename(columns={"stake_id": "Stake", "HW": "stake_height_cm", "DW": "snow_depth_cm"})
    stakes_w["surface"] = "s"
    stakes_w["date"] = date

    stakes_s = stakes[["Stake", "HL", "Surface", "geometry"]].rename(columns={"stake_id": "Stake", "HL": "stake_height_cm", "Surface": "surface"})
    stakes_s["snow_depth_cm"] = 0.0
    # TODO: Try to find this information instead of guessing on September
    stakes_s["date"] = datetime.date(year, 9, 1)


    return probings, stakes, densities

def parse_2016_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Densities are made first (in 2016, only the densities have a winter date)
    densities = data["Density"]
    densities = set_header(densities, ["From", "To", "Density"])
    densities = densities[densities["Density"].apply(lambda s: is_number(s))]
    if densities.shape[0] > 0:
        densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["From"], densities["To"])
        densities = densities[["depth_cm", "Density"]].rename(columns={"Density": "density"})

    probings = set_header(data["Probing data"], column_values=["Easting", "Northing"])
    probings.rename(columns={"#": "stake_id", "Snow depth": "snow_depth_cm", "Note": "note"}, inplace=True)
    probings = probings[probings["snow_depth_cm"].apply(lambda s: is_number(s))]

    # If there are two easting columns, there are coordinates in both RT90 and SWEREF99TM
    if len(list(filter(lambda col: col == "Easting", probings.columns))) > 1:
        probings["geometry"] = gpd.points_from_xy(probings.iloc[:, 3], probings.iloc[:, 4], crs=3006)
    else:
        probings["geometry"] = gpd.points_from_xy(probings["Easting"], probings["Northing"], crs=3022).to_crs(3006)
    probings = probings[["stake_id", "snow_depth_cm", "note", "geometry"]]

    try:
        date = densities.iloc[:, -1][densities.iloc[:, -1].apply(lambda d: hasattr(d, "year"))].iloc[0].date()
    except IndexError:
        data_str = data["Probing data"].astype(str).applymap(lambda s: s.lower())

        potential_dates = data_str.values[data_str.applymap(lambda s: "apr" in s).values | data["Probing data"].applymap(lambda d: isinstance(d, datetime.datetime))]

        potential_dates = np.append(potential_dates, data["Density"].values[data["Density"].applymap(lambda d: isinstance(d, datetime.datetime))])
        date = None
        for potential_date in potential_dates:
            try:
                date = pd.Timestamp(potential_date).date()
            except ValueError:
                continue
            break

        if date is None:
            raise ValueError("Could not find winter date")
        
    probings["date"] = date
    densities["date"] = date
    
    stakes_raw = data["Stake ablation"]
    stakes = pd.DataFrame()


    mask = np.sum([stakes_raw == value for value in ["Stake", "Easting", "Northing", "HW"]], axis=0).sum(axis=1) >= 4

    col_index = np.argwhere(mask).ravel()[-1]
    # Set the column row to the columns
    stakes_raw.columns = stakes_raw.loc[col_index]
    stakes_raw = stakes_raw.dropna(how="all")

    stakes = stakes_raw[["Stake","Easting", "Northing", "HW", "DW"]].copy().rename(columns={"HW": "height", "DW": "winter_snow_depth"})
    stakes["surface"] = "S"
    # Only keep the data after the column row.
    stakes = stakes.loc[:, ~stakes.columns.duplicated()]
    stakes = stakes[stakes["Stake"] != "#"]
    stakes = stakes.loc[col_index + 1 :]

    stakes["date"] = date

    previous_date = pd.Timestamp(year=1900, month=1, day=1)
    for i, col in enumerate(stakes_raw):
        if "H" not in str(col):
            continue
        if not col.replace("H", "").isnumeric() and col != "HS":
            continue
        subset = stakes_raw.iloc[:, [i, i+1]].copy()
        subset["Stake"] = stakes_raw.iloc[:, list(stakes_raw.columns).index("Stake")]

        # The datetimes are all over the place (and sometimes incorrect), so manual parsing is better
        date_str = str(subset.iloc[2, 1])
        if date_str == "nan":
            continue
        month = date_str[5:7]
        day = date_str[8:10]
        date = pd.Timestamp(year=year, month=int(month), day=int(day))
        
        if date < previous_date and col != "HS":
            date = pd.Timestamp(year=date.year + 1, month=date.month, day=date.day)

        previous_date = date

        for j, row in subset.iterrows():
            if all((any("(m)" == row.values), any("(S/SI/I/F)" == row.values))):
                data_header_row = j
                break
        subset = subset.iloc[data_header_row + 1:]
        # Remove rows where no stake height exists
        subset = subset[~subset.iloc[:, 0].isna()]
        subset.columns = ["height", "surface", "Stake"]
        subset["date"] = date

        stakes = pd.concat([stakes, subset], ignore_index=True)


    stakes = stakes[stakes["height"].apply(lambda s: is_number(s))]
    stakes.rename(columns={"Stake": "stake_id"}, inplace=True)
    stakes["stake_height_cm"] = stakes["height"] * 100
    stakes["snow_depth_cm"] = stakes["winter_snow_depth"] * 100

    return probings, stakes, densities


def parse_2017_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    probings = data["Mårma Sondering"]

    date_str = str(probings.iloc[1, 0])
    date = datetime.date(year, int(date_str[:2]), int(date_str[2:]))

    probings = set_header(probings, ["Date", "Punkt"])
    probings.columns = ["date", "probe_point_id", "snow_depth_cm", "note"]
    probings["date"] = date

    stakes = set_header(data["Mårma stakes"], ["Date", "ID", "Snow Depth"])
    columns = {
        "Date": "date",
        "ID": "stake_id",
        "Snow Depth": "snow_depth_cm",
        "Stake Height": "stake_height_cm",
        "Note": "note",
    }
    stakes = stakes.rename(columns=columns)[list(columns.values())].dropna(how="all")

    densities = data["Mårma Densitet"]
    date_str = str(densities.iloc[1, 0])
    date = datetime.date(year, int(date_str[:2]), int(date_str[2:]))
    densities = set_header(densities, ["Date", "Från (cm)", "Till (cm)"])
    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["Från (cm)"], densities["Till (cm)"])
    densities = densities[["depth_cm", "Densitet"]].rename(columns={"Densitet": "density"})
    densities["date"] = date

    return probings, stakes, densities


def parse_2019_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probings = data["Mårma_PROBE"]

    date = (
        probings.iloc[0, 1].date()
        if isinstance(probings.iloc[0, 1], datetime.datetime)
        else datetime.date(int(year), 4, 1)
    )

    probings = set_header(probings, ["E", "N"])
    probings["geometry"] = gpd.points_from_xy(probings["E"], probings["N"])
    probings = (
        probings.rename(
            columns={"SNOW DEPTH (cm)": "snow_depth_cm", "PROBE POINT ID": "probe_point_id", "NOTE": "note"}
        )
        .drop(columns=["#", "E", "N"])
        .dropna(subset=["snow_depth_cm"])
    )
    probings["date"] = date

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

    densities = data["Mårma_DENSITY"]

    density_date = densities.iloc[0, 1] if isinstance(densities.iloc[0, 1], datetime.datetime) else date
    densities = set_header(densities, ["FROM (cm)", "TO (cm)"])

    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["FROM (cm)"], densities["TO (cm)"])
    densities = densities.rename(columns={"DENSITY": "density"})[["depth_cm", "density"]]
    densities["date"] = density_date

    return probings, stakes, densities


def is_number(obj: object) -> bool:
    obj = str(obj)

    try:
        number = float(obj)
        return not np.isnan(number)
    except ValueError:
        return False


def set_header(dataframe: pd.DataFrame, column_values: list[str]) -> pd.DataFrame:
    new_data = dataframe.copy()

    mask = np.sum([dataframe == value for value in column_values], axis=0).sum(axis=1) >= len(column_values)

    header_index = np.argwhere(mask).ravel()[0]

    new_data.columns = new_data.iloc[header_index, :]
    new_data = new_data.iloc[header_index + 1 :, :]

    return new_data
