from __future__ import annotations

import datetime
import os
import re
import json
import warnings
import io
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
        all_files = os.listdir(base_directory.joinpath(year))

        xls_files = list(filter(lambda s: "xls" in s and "lock" not in s, all_files))
        txt_files = list(filter(lambda s: any(ext in s for ext in ["txt", "csv", "geojson"]) and "lock" not in s, all_files))
        if len(xls_files) == 0 and len(txt_files) == 0:
            continue

        with warnings.catch_warnings():

            warnings.filterwarnings("ignore", "Data Validation extension is not supported")
            data: dict[str, pd.DataFrame] = {}
            i = 0
            for filename in xls_files:
                for key, value in pd.read_excel(
                    base_directory.joinpath(year).joinpath(filename), sheet_name=None, header=None
                ).items():
                    key_to_use = key
                    for i in range(1, 100):
                        if key_to_use in data:
                            key_to_use = key + f"_{i}"
                        else:
                            break

                    data[key_to_use] = value

            for filename in txt_files:
                with open(base_directory.joinpath(year).joinpath(filename)) as infile:
                    data[os.path.splitext(filename)[0]] = infile.read().splitlines()

        probings, stakes, densities = figure_out_version(data, int(year))

        stakes = stakes.loc[:, ~stakes.columns.duplicated()]

        probing_dfs.append(probings)
        stake_dfs.append(stakes)
        density_dfs.append(densities)

    densities = pd.concat(density_dfs, ignore_index=True)
    probings = pd.concat(probing_dfs, ignore_index=True)[["stake_id", "snow_depth_cm", "date", "geometry", "parser"]]
    stakes = pd.concat(stake_dfs, ignore_index=True)[
        ["stake_id", "stake_height_cm", "snow_depth_cm", "note", "surface", "date", "geometry", "parser"]
    ]

    print(probings.loc[probings["date"].apply(lambda x: isinstance(x, str))])
    stakes["snow_depth_cm"] = stakes["snow_depth_cm"].fillna(0)
    stakes = stakes[is_number(stakes["stake_height_cm"])]
    for dataframe in [stakes, probings]:
        dataframe["date"] = dataframe["date"].apply(lambda d: d if isinstance(d, datetime.date) else d.date())

    stakes.loc[stakes["geometry"].apply(lambda p: ~np.isfinite(p.x) if p is not None else True), "geometry"] = pd.NA


    for dataframe in [stakes, probings]:
        for i, point in dataframe.loc[dataframe["geometry"].isna()].iterrows():
            other_measurements = dataframe.loc[
                (dataframe["stake_id"] == point["stake_id"]) & (~dataframe["geometry"].isna()), "geometry"
            ]
            if np.size(other_measurements) == 0:
                continue
            dataframe.loc[i, "geometry"] = other_measurements.iloc[0]
    stakes = gpd.GeoDataFrame(stakes, crs=3006)
    probings = gpd.GeoDataFrame(probings, crs=3006).sort_values("date")

    return probings, stakes, densities


def figure_out_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    try:
        if all(key in data for key in ["densities_digitized", "probings_digitized", "stakes_digitized"]):
            return parse_1998_version(data, year)
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


def parse_1998_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probings = gpd.GeoDataFrame.from_features(json.loads("\n".join(data["probings_digitized"])))

    stakes = gpd.GeoDataFrame.from_features(json.loads("\n".join(data["stakes_digitized"])))
    stakes.rename(columns={"height": "stake_height_cm", "snow_depth": "snow_depth_cm"}, inplace=True)

    densities = pd.read_csv(io.StringIO("\n".join(data["densities_digitized"])))
    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["from"], densities["to"])
    densities["geometry"] = pd.NA

    for dataframe in [densities, probings, stakes]:
        dataframe["date"] = pd.to_datetime(dataframe["date"])

    for i, row in densities.iterrows():
        if pd.isna(row["note"]) or "by stake " not in row["note"].lower():
            continue
        nearby_stake = row["note"][row["note"].lower().index("by stake") + 8:].strip()
        if " " in nearby_stake:
            nearby_stake = nearby_stake[:nearby_stake.index(" ")]
        if "." in nearby_stake:
            nearby_stake = nearby_stake[:nearby_stake.index(".")]

        try:
            densities.loc[i, "geometry"] = stakes.loc[stakes["stake_id"] == nearby_stake, "geometry"].iloc[0]
        except IndexError as exception:
            raise ValueError(f"Could not find stake {nearby_stake} in {stakes['stake_id'].unique()}") from exception

    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "1998_version"
    return probings, stakes, densities

def parse_1999_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    winter_date = datetime.date(year=year, month=5, day=1)
    summer_date = datetime.date(year=year, month=9, day=1)

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
    densities["date"] = winter_date

    probings = set_header(data["AB"], ["cm w eq"]).iloc[:, :3]
    probings.columns = ["X", "Y", "snow_depth_cm"]
    probings = probings[is_number(probings["snow_depth_cm"])]

    probings["geometry"] = parse_coordinates(probings["X"], probings["Y"])
    probings["date"] = winter_date


    stakes_raw = set_header(data["Stakar vår"], ["X", "Y", "Djup"])
    stakes_raw = stakes_raw[is_number(stakes_raw["Höjd"])]
    stakes_raw["geometry"] = parse_coordinates(stakes_raw["X"], stakes_raw["Y"])

    stakes_raw["stake_id"] = "M" + stakes_raw.index.astype(str)

    stakes_w = stakes_raw.rename(columns={"Höjd": "stake_height_cm", "Djup": "snow_depth_cm"})[["stake_id", "stake_height_cm", "snow_depth_cm", "geometry"]]
    stakes_w["date"] = winter_date

    stakes_s = stakes_raw.rename(columns={"Höjd höst": "stake_height_cm"})[["stake_id", "stake_height_cm", "geometry"]]
    stakes_s["snow_depth_cm"] = 0
    stakes_s["date"] = summer_date

    stakes = pd.concat([stakes_w, stakes_s], ignore_index=True)
    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "1999_version"
    return probings, stakes, densities


def parse_2001_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    winter_date = datetime.date(year=year, month=5, day=1)
    summer_date = datetime.date(year=year, month=9, day=1)

    probings = set_header(data["Sond_koord"], ["X", "Y"])
    probings.rename(columns={"Sondvärde": "snow_depth_cm"}, inplace=True)
    probings["geometry"] = parse_coordinates(probings["X"], probings["Y"])
    probings["date"] = winter_date

    stakes_raw = set_header(data["stake_koord"], ["X", "Y"])
    cols = [str(col).strip() for col in stakes_raw.columns]
    cols[cols.index("Vinter") + 1] = "snow_depth_cm"
    
    for i in range(stakes_raw.shape[1]):
        if len(str(stakes_raw.iloc[:, i]).split("-")) >= 3:
            cols[i] = "stake_id"
            break
    stakes_raw.columns = cols

    stakes_raw = stakes_raw[is_number(stakes_raw["Vinter"])]
    stakes_raw["geometry"] = parse_coordinates(stakes_raw["X"], stakes_raw["Y"])

    stakes_w = stakes_raw.rename(columns={"Vinter": "stake_height_cm"})
    stakes_w["date"] = winter_date

    stakes_s = stakes_raw.rename(columns={"Sommar": "stake_height_cm"})
    stakes_s["date"] = summer_date
    stakes_s["snow_depth_cm"] = 0
    stakes_s.index += 1000

    stake_cols = ["stake_id", "stake_height_cm", "snow_depth_cm", "date", "geometry"]
    stakes = pd.concat([stakes_w[stake_cols], stakes_s[stake_cols]])

    densities = set_header(data["Blad1"], ["Djup", "Vikt"])
    densities = densities[is_number(densities["Djup"])]

    densities["Längd"] = densities["Längd"].fillna(25)

    densities["depth_cm"] = pd.IntervalIndex.from_arrays(left=densities["Djup"] - densities["Längd"], right=densities["Djup"])

    densities = densities.rename(columns={"Densitet": "density"})[["depth_cm", "density"]]
    densities["date"] = winter_date
    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2001_version"

    return probings, stakes, densities

def parse_2002_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    probings = set_header(data["marma2002"], ["Date", "Sounding"])
    probings.columns = [str(col).strip() for col in probings.columns]
    probings["snow_depth_cm"] = probings["Sounding"] * 100
    probings["geometry"] = parse_coordinates(probings["Easting"], probings["Northing"])
    probings["note"] = "Prober: " + probings["Sounder"] + ". Alternative: "  + (probings["Alternative"] * 100).astype(str)

    probings = gpd.GeoDataFrame(probings.rename(columns={"Date": "date"})[["snow_depth_cm", "date", "note", "geometry"]], crs=3006)
    probings["date"] = probings["date"].ffill()

    stakes_raw = set_header(data["Blad1"], ["Date", "Height", "Sounder"])
    cols = [str(col).strip() for col in stakes_raw.columns]
    cols[list(data["Blad1"].iloc[0].values).index("VT-02") + 1] = "spring_height"
    cols[list(data["Blad1"].iloc[0].values).index("HT-02")] = "fall_height"
    stakes_raw.columns = cols

    stakes_raw["note"] = "Prober: " + stakes_raw["Sounder"].astype(str)
    stakes_raw["geometry"] = parse_coordinates(stakes_raw["Easting"], stakes_raw["Northing"])

    stakes = stakes_raw.rename(columns={"Date": "date", "WPT": "stake_id", "spring_height": "stake_height_cm", "d": "snow_depth_cm"})[["stake_id", "date", "stake_height_cm", "snow_depth_cm", "note", "geometry"]]

    stakes_s = stakes_raw.rename(columns={"WPT": "stake_id", "fall_height": "stake_height_cm"})[["stake_id", "stake_height_cm", "geometry"]]
    stakes_s[["date", "note", "snow_depth_cm"]] = datetime.date(year=year, month=9, day=1), pd.NA, 0

    stakes = stakes.append(stakes_s, ignore_index=True)
    stakes = stakes[~stakes["stake_id"].isna()]

    density_df_breaks = pd.DataFrame(columns=["start"])
    density_coordinates = pd.DataFrame(columns=["easting", "northing"])
    density_i = 0
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
    
    density_coordinates = parse_coordinates(density_coordinates["easting"].values, density_coordinates["northing"].values)
    density_dfs: list[pd.DataFrame] = []

    for i, row in density_df_breaks.iterrows():
        density_df = set_header(data["Densitet"].iloc[row["start"]:row["end"]], ["Djup (cm)"])
        density_df = density_df[~density_df["Densitet (g/cm3)"].isna()]

        if "Bottendjup (cm)" in density_df:
            density_df["depth_cm"] = pd.IntervalIndex.from_arrays(left=density_df["Topp Djup (cm)"], right=density_df["Bottendjup (cm)"])
        else:
            density_df["depth_cm"] = pd.IntervalIndex.from_arrays(left=density_df["Djup (cm)"] - density_df["Längd snödel (cm)"], right=density_df["Djup (cm)"])


        density_df["date"] = probings["date"].max()
        density_df["geometry"] = density_coordinates[i]
        density_dfs.append(density_df.rename(columns={"Densitet (g/cm3)": "density"})[["depth_cm", "density", "date", "geometry"]])

    densities = pd.concat(density_dfs, ignore_index=True)
    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2002_version"

    return probings, stakes, densities

def parse_2003_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    probings = data["sondering"]
    date_str = probings.iloc[0, 1]
    probings = set_header(probings, ["wpt", "d"])
    probings["wpt"] = probings["wpt"].astype(int)

    gps_coords = gpd.GeoDataFrame()
    for row in data["marma03wpt"]:

        try:
            wpt_name, date_str, time_str, easting, northing = re.split("[\s]+", row.replace("\t", " "))[:5]
        except ValueError as exception:
            if "not enough values to unpack" not in str(exception):
                raise exception
            continue

        if not is_number(wpt_name):
            continue

        date = datetime.datetime.strptime(date_str + " " + time_str, "%d-%b-%y %H:%M")
        gps_coords.loc[gps_coords.shape[0], ["wpt", "date", "easting", "northing"]] = (
            wpt_name,
            date,
            float(easting) * 1e3,
            float(northing) * 1e3,
        )

    gps_coords["geometry"] = parse_coordinates(gps_coords["easting"], gps_coords["northing"])
    gps_coords["wpt"] = gps_coords["wpt"].astype(int)

    probings = probings.merge(gps_coords, on="wpt").rename(
        columns={"d": "snow_depth_cm", "kommentar (för orientering, mest)": "note"}
    )

    probings = probings[is_number(probings["snow_depth_cm"])]

    probings = probings[["snow_depth_cm", "note", "date", "geometry"]]

    # Some dates are most likely wrong; they are from 2002 supposedly, which doesn't make sense as another...
    # ... document says these are from April 2003!
    probings.loc[probings["date"].apply(lambda d: d.year == 2002), "date"] = pd.NA
    probings["date"] = probings["date"].bfill()

    stakes = gpd.GeoDataFrame(
        columns=["stake_id", "snow_depth_cm", "stake_height_cm", "date", "note", "geometry"], crs=3006
    )

    unknowns_n = 0
    for _, probing in probings.iterrows():
        if not any(s in str(probing["note"]) for s in ["h=", "h:"]):
            continue
        stake_height_cm = np.nan
        stake_id = ""

        for word in str(probing["note"]).split(" "):
            if len(word) == 4 and "-" in word:
                stake_id = word
            elif "h:" in word or "h=" in word:
                stake_height_cm = int(word.replace("h:", "").replace("h=", "").replace(",", ""))
                # Ugly hardcoding of note ("300 was removed")
                if "300 togs av" in probing["note"]:
                    stake_height_cm -= 300
        if len(stake_id) == 0:
            stake_id = f"unknown_{unknowns_n}"
            unknowns_n += 1

        stakes.loc[stakes.shape[0]] = (
            stake_id,
            probing["snow_depth_cm"],
            stake_height_cm,
            probing["date"],
            probing["note"],
            probing["geometry"],
        )

    stakes["x"] = stakes["geometry"].apply(lambda geom: geom.x)

    # Extract all of the stake names (sorted by the Easting coordinate)
    sorted_stake_names = [name for name in stakes.sort_values("x")["stake_id"].values if "unknown" not in name]
    stakes.drop(columns="x", inplace=True)
    # There are no stake measurements anywhere from the summer, only the calculated ablation...
    summer_ablation = set_header(data["stake_koord"], ["X", "Y", "Ablation"]).sort_values("X")
    # We assume that these are the same as in the "stake_koord" sheet
    assert len(sorted_stake_names) == summer_ablation.shape[0]
    summer_ablation["stake_id"] = sorted_stake_names

    # density_dfs = pd.DataFrame(columns=["location", "depth_cm", "density", "date"])
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

    summer_ablation["date"] = datetime.date(year=year, month=9, day=1)
    summer_ablation[
        "note"
    ] = "Only w.e. ablation values were available so the stake height is derived from this and the winter values. A compacted snow density of 0.6 was assumed, and 0.9 for pure ice."

    stakes = stakes.append(
        summer_ablation.rename(columns={"summer_snow_w_abl": "snow_depth_cm", "summer_height": "stake_height_cm"})[
            stakes.columns
        ]
    )
    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2003_version"

    return probings, stakes, densities


def parse_2004_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

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
    stakes_w["date"] = datetime.date(year, 4, 15)  # TODO: Find the actual date

    stakes_s = stakes.rename(columns={"sommarhöjd": "stake_height_cm"})
    stakes_s["snow_depth_cm"] = 0
    stakes_s["date"] = datetime.date(year, 9, 1)  # TODO: Find the actual date

    stakes = pd.concat([stakes_w, stakes_s])[["stake_id", "snow_depth_cm", "stake_height_cm", "date", "geometry"]]

    densities = set_header(data["Densitet"], ["Snödjup", "Densitet (kg/m3)"]).rename(
        columns={"Densitet (kg/m3)": "density"}
    )
    densities = densities[is_number(densities["density"])]
    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities.iloc[:, 1], densities["Snödjup"])
    densities = densities[~densities["depth_cm"].isna()]
    densities["date"] = probings["date"].iloc[0].date()

    densities = densities[["depth_cm", "density", "date"]]

    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2004_version"

    return probings, stakes, densities


def parse_2005_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    probings = set_header(data["sond"], ["Alt", "Snödjup (m)"])
    cols = [str(col).strip() for col in probings.columns]
    cols[-1] = "note"
    probings.columns = cols

    # TODO: Find out the actual date instead of guessing.
    date = datetime.date(year, 4, 15)

    probings.loc[~probings["Alt"].isna(), "Snödjup (m)"] *= 100
    probings["geometry"] = parse_coordinates(probings["Easting"], probings["Northing"])

    probings = probings[["Snödjup (m)", "note", "geometry"]]
    probings.columns = ["snow_depth_cm", "note", "geometry"]
    probings["date"] = date

    densities = set_header(data["Density"], ["From", "To", "Density"])
    densities = densities[is_number(densities["Density"])]
    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["From"], densities["To"])
    densities = densities[["depth_cm", "Density"]].rename(columns={"Density": "density"})
    densities["date"] = date

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
    stakes_w["date"] = date

    stakes_s = stakes[["Stake", "H1", "geometry"]].rename(columns={"Stake": "stake_id", "H1": "stake_height_cm"})
    stakes_s["snow_depth_cm"] = 0.0
    # TODO: Find the actual date
    stakes_s["date"] = datetime.date(year, 9, 1)

    stakes = pd.concat([stakes_w, stakes_s], ignore_index=True)

    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2005_version"
    return probings, stakes, densities


def parse_2013_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    densities = set_header(data["Density"], ["From", "To", "Density"])
    date = densities.iloc[0, -1].date()
    densities = densities[densities["Density"].apply(lambda s: is_number(s) and float(s) != 0.0)]
    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["From"], densities["To"])
    densities = densities[["depth_cm", "Density"]].rename(columns={"Density": "density"})
    densities["date"] = date

    probings = set_header(data["Probing data"], ["Easting", "Northing"])
    probings = probings[is_number(probings["Snow depth"])]
    probings["geometry"] = parse_coordinates(probings["Easting"], probings["Northing"])

    probings = probings[["Snow depth", "Note", "geometry"]]
    probings.columns = ["snow_depth_cm", "note", "geometry"]
    probings["date"] = date

    stakes = set_header(data["Summer balance calc"], ["Stake", "Easting", "Northing", "Elevation"])
    stakes = stakes[is_number(stakes["HW"])]
    stakes["geometry"] = parse_coordinates(stakes["Easting"], stakes["Northing"])

    stakes_w = stakes[["Stake", "HW", "DW", "geometry"]].rename(
        columns={"stake_id": "Stake", "HW": "stake_height_cm", "DW": "snow_depth_cm"}
    )
    stakes_w["surface"] = "s"
    stakes_w["date"] = date

    stakes_s = stakes[["Stake", "HL", "Surface", "geometry"]].rename(
        columns={"stake_id": "Stake", "HL": "stake_height_cm", "Surface": "surface"}
    )
    stakes_s["snow_depth_cm"] = 0.0
    # TODO: Try to find this information instead of guessing on September
    stakes_s["date"] = datetime.date(year, 9, 1)

    stakes = pd.concat([stakes_w, stakes_s])

    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2013_version"
    return probings, stakes, densities


def parse_2016_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Densities are made first (in 2016, only the densities have a winter date)
    densities = data["Density"]
    densities = set_header(densities, ["From", "To", "Density"])
    densities = densities[is_number(densities["Density"])]
    if densities.shape[0] > 0:
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

    try:
        date = densities.iloc[:, -1][densities.iloc[:, -1].apply(lambda d: hasattr(d, "year"))].iloc[0].date()
    except IndexError:
        data_str = data["Probing data"].astype(str).applymap(lambda s: s.lower())

        potential_dates = data_str.values[
            data_str.applymap(lambda s: "apr" in s).values
            | data["Probing data"].applymap(lambda d: isinstance(d, datetime.datetime))
        ]

        potential_dates = np.append(
            potential_dates,
            data["Density"].values[data["Density"].applymap(lambda d: isinstance(d, datetime.datetime))],
        )
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

    stakes["date"] = date

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
                date = datetime.date(year, 9, 1)
            else:
                continue
        else:
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
        subset = subset.iloc[data_header_row + 1 :]
        # Remove rows where no stake height exists
        subset = subset[~subset.iloc[:, 0].isna()]
        subset.columns = ["height", "surface", "Stake"]
        subset["date"] = date

        stakes = pd.concat([stakes, subset], ignore_index=True)

    stakes = stakes[is_number(stakes["height"])]
    stakes.rename(columns={"Stake": "stake_id"}, inplace=True)
    stakes["stake_height_cm"] = stakes["height"] * 100
    stakes["snow_depth_cm"] = stakes["winter_snow_depth"] * 100

    stakes["geometry"] = parse_coordinates(
        stakes.loc[:, stakes.columns == "Easting"].iloc[:, -1], stakes.loc[:, stakes.columns == "Northing"].iloc[:, -1]
    )
    stakes["stake_id"] = stakes["stake_id"].astype(str).str.replace("M-", "M")

    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2016_version"
    return probings, stakes, densities


def parse_2017_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    probings = data["Mårma Sondering"]

    date_str = str(probings.iloc[1, 0])
    date = datetime.date(year, int(date_str[:2]), int(date_str[2:]))

    probings = set_header(probings, ["Date", "Punkt"])
    probings.columns = ["date", "stake_id", "snow_depth_cm", "note"]
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

    ##stakes["stake_id"] = "M" + stakes["stake_id"].astype(str)

    stakes = stakes[list(columns.values())]
    stakes[~is_number(stakes["snow_depth_cm"])] = np.nan

    densities = data["Mårma Densitet"]
    date_str = str(densities.iloc[1, 0])
    date = datetime.date(year, int(date_str[:2]), int(date_str[2:]))
    densities = set_header(densities, ["Date", "Från (cm)", "Till (cm)"])
    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["Från (cm)"], densities["Till (cm)"])
    densities = densities[["depth_cm", "Densitet"]].rename(columns={"Densitet": "density"})
    densities["date"] = date

    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2017_version"
    return probings, stakes, densities


def parse_2019_version(data: dict[str, pd.DataFrame], year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probings = data["Mårma_PROBE"]

    date = (
        probings.iloc[0, 1].date()
        if isinstance(probings.iloc[0, 1], datetime.datetime)
        else datetime.date(int(year), 4, 1)
    )

    probings = set_header(probings, ["E", "N"])
    probings["geometry"] = parse_coordinates(probings["E"], probings["N"])
    probings = (
        probings.rename(columns={"SNOW DEPTH (cm)": "snow_depth_cm", "PROBE POINT ID": "stake_id", "NOTE": "note"})
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
    stakes = stakes[["stake_id", "stake_height_cm", "snow_depth_cm", "note", "surface", "date"]]

    densities = data["Mårma_DENSITY"]

    density_date = densities.iloc[0, 1] if isinstance(densities.iloc[0, 1], datetime.datetime) else date
    densities = set_header(densities, ["FROM (cm)", "TO (cm)"])

    densities["depth_cm"] = pd.IntervalIndex.from_arrays(densities["FROM (cm)"], densities["TO (cm)"])
    densities = densities.rename(columns={"DENSITY": "density"})[["depth_cm", "density"]]
    densities["date"] = density_date

    for dataframe in probings, stakes, densities:
        dataframe["parser"] = "2019_version"
    return probings, stakes, densities


@np.vectorize
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


def parse_coordinates(easting: np.ndarray, northing: np.ndarray) -> gpd.GeoSeries:
    """
    Parse arrays of coordinates.

    They are assumed to be either SWEREF99TM or RT90 2.5gon V, depending on the size of the easting coord.

    :param easting: An array-like of easting coordinates.
    :param northing: An array-like of northing coordinates.

    :returns: A GeoSeries in SWEREF99TM
    """
    easting = np.asarray(easting)
    northing = np.asarray(northing)

    if easting[0] < 1e4:
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


def test_is_number():
    assert is_number("9")
    assert not is_number("jgh")

    assert np.array_equal(is_number(["9", "ladw"]), [True, False])
