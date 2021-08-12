from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xdem


def plot_variograms(ddems: list[xdem.dDEM]):

    for i, ddem in enumerate(ddems):
        year_label = f"{ddem.start_time.year} to {ddem.end_time.year}"
        ax0 = plt.subplot(3, len(ddems), i + 1)
        plt.title(year_label)
        ax1 = plt.subplot(3, len(ddems), len(ddems) + i + 1)
        ax2 = plt.subplot(3, len(ddems), 2 * len(ddems) + i + 1)

        variogram = ddem.variograms["variogram"]
        vgm_model = ddem.variograms["vgm_model"]
        params = ddem.variograms["vgm_params"]
        empirical = ddem.variograms["empirical_variance"]
        standardized_std = ddem.variograms["standardized_std"]

        # Plot the histogram manually with fill_between
        interval_var = [0] + list(variogram.bins)
        for i in range(len(variogram)):
            count = variogram["count"].values[i]
            ax0.fill_between(
                [interval_var[i], interval_var[i + 1]],
                [0] * 2,
                [count] * 2,
                facecolor=plt.cm.Greys(0.75),
                alpha=1,
                edgecolor="white",
                linewidth=0.5,
            )

        model_xs = np.linspace(0, np.max(variogram["bins"]), 10000)

        ax1.plot(model_xs, vgm_model(model_xs) * standardized_std)

        # Get the bins center
        bins_center = np.subtract(variogram["bins"], np.diff([0] + variogram["bins"].tolist()) / 2)

        # If all the estimated errors are all NaN (single run), simply plot the empirical variogram
        if np.all(np.isnan(variogram["exp"])):
            ax1.scatter(
                bins_center, variogram["exp"] * standardized_std, label="Empirical variogram", color="blue", marker="x"
            )
        # Otherwise, plot the error estimates through multiple runs
        else:
            ax1.errorbar(
                bins_center,
                variogram["exp"] * standardized_std,
                yerr=variogram["err_exp"],
                label="Empirical variogram (1-sigma s.d)",
                fmt="x",
            )

        for axis in [ax0, ax1]:
            axis.set_xscale("log")
            axis.set_xlim(5, variogram["bins"].max())
            axis.set_xlabel("Lag (m)")

        areas = np.linspace(np.prod(ddem.res), np.prod(ddem.res) * np.prod(ddem.shape), 1000)

        estimated_error: list[float] = []
        for area in areas:
            neff_doublerange = xdem.spatialstats.neff_circ(
                area, [(params[0], "Sph", params[1]), (params[2], "Sph", params[3])]
            )
            estimated_error.append(
                xdem.spatial_tools.nmad(ddem.data[ddem.stable_terrain_mask]) / np.sqrt(neff_doublerange)
            )

        ax2.plot(areas, estimated_error)
        ax2.scatter(empirical.index, empirical["std"])
        ax2.set_xscale("log")
        ax2.set_xlabel("Averaging area (m²)")
        ax2.set_ylabel("Elevation uncertainty (std) (m)")
        ax2.set_yscale("logit")

        # xdem.spatialstats.plot_vgm(variogram, xscale_range_split=[100, 1000, 10000], list_fit_fun=[vgm_model],
        #                   list_fit_fun_label=['Standardized double-range variogram'], ax=ax0)
        """
        variogram = ddem.variance.copy()
        variogram[["nmad", "upper_quartile", "lower_quartile"]] = np.sqrt(variogram["nmad"].max() - variogram[["nmad", "upper_quartile", "lower_quartile"]])

        plt.errorbar(
            variogram["radius"].values,
            variogram["nmad"],
            yerr=np.abs(ddem.variance[["upper_quartile", "lower_quartile"]].values.T - ddem.variance["nmad"].values),
            color="orange",
        )
        plt.fill_between(variogram["radius"], variogram["upper_quartile"], variogram["lower_quartile"], color="orange", alpha=0.3)
        plt.plot(variogram["radius"], variogram["nmad"], color="orange")
        plt.scatter(variogram["radius"], variogram["nmad"], color="orange", s=1)

        plt.xlabel("Lag (m)")
        plt.xscale("log")
        plt.ylabel("NMAD (m)")
        #plt.ylim(0, 1.5)

        plt.subplot(2, len(ddems), len(ddems) + i + 1)

        plt.errorbar(
            ddem.variance.index,
            ddem.variance["nmad"],
            yerr=np.abs(
                ddem.variance[["lower_quartile", "upper_quartile"]].values.T - ddem.variance["nmad"].values
            ),
        )
        plt.scatter(ddem.variance.index, ddem.variance["nmad"], s=1)

        plt.xlabel("Integrated area (m²)")
        plt.xscale("log")
        plt.ylabel("NMAD (m)")
        #plt.ylim(-0.02, 0.4)
        """
    plt.show()
