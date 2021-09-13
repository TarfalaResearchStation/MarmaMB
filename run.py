import datetime
import pathlib

import matplotlib.pyplot as plt
import numpy as np
#import xdem

import marma.mb_parsing

import pickle
import pdb


def main():

    
    mb = marma.mb_parsing.read_all_data(pathlib.Path("input/massbalance/"))

    return
    reference_year = 2016

    dems, ddems, unstable_terrain = marma.main.prepare_dems(reference_year=reference_year)
    
    changes = marma.analysis.volume_change(ddems, unstable_terrain)

    marma.plotting.plot_volume_change(changes)

    return
    
    marma.plotting.plot_variograms(ddems)
    plt.show()
    return


    fig = plt.figure()
    for i, ddem in enumerate(ddems):

        ddem.data /= abs(ddem.start_time.year - ddem.end_time.year)
        axis = plt.subplot(2, 5, i + 1)
        ddem.show(
            ax=axis,
            cmap="RdBu",
            vmin=-2,
            vmax=2,
            title=f"{min(ddem.start_time.year, ddem.end_time.year)} -- {max(ddem.start_time.year, ddem.end_time.year)}",
            add_cb=False
        )
        axis = plt.subplot(2, 5, i + 6)
        xdem.DEM.from_array(ddem.error, ddem.transform, ddem.crs, -9999).show(cmap="Reds", ax=axis, vmin=0, vmax=10, add_cb=False)

    plt.show()


if __name__ == "__main__":
    main()
