import glob
import pkg_resources
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles
from pydmc.dmcplot import DmcPlot


def flanker_data():
    """Load raw Flanker data from Ulrich et al. (2015)."""
    datafile = pkg_resources.resource_stream(__name__, "data/flankerData.csv")
    return pd.read_csv(datafile, sep="\t")


def simon_data():
    """Load raw Simon task data from Ulrich et al. (2015)."""
    datafile = pkg_resources.resource_stream(__name__, "data/simonData.csv")
    return pd.read_csv(datafile, sep="\t")


class DmcOb:
    def __init__(
        self,
        data,
        n_caf=5,
        n_delta=19,
        p_delta=None,
        t_delta=1,
        outlier=(200, 1200),
        columns=("Subject", "Comp", "RT", "Error"),
        comp_coding=("comp", "incomp"),
        error_coding=(0, 1),
        sep="\t",
        skiprows=0,
    ):
        """

        Parameters
        ----------
        data
        n_caf
        n_delta
        p_delta
        t_delta
        outlier
        columns
        comp_coding
        error_coding
        sep
        skiprows
        """
        self.n_caf = n_caf
        self.n_delta = n_delta
        self.p_delta = p_delta
        self.t_delta = t_delta
        self.outlier = outlier
        self.columns = columns
        self.comp_coding = comp_coding
        self.error_coding = error_coding

        if not isinstance(data, pd.DataFrame):
            self.data = self.read_data_files(data, sep=sep, skiprows=skiprows)
            self._columns()
            self._comp_coding()
            self._error_coding()
            self._outlier()
        else:
            self.data = data
            self._columns()
            self._comp_coding()
            self._error_coding()
            self._outlier()

        self._aggregate_trials()
        self._aggregate_subjects()
        self._calc_caf_values()
        self._calc_delta_values()

    @staticmethod
    def read_data_files(data, sep="\t", skiprows=0):
        """

        Parameters
        ----------
        data
        sep
        skiprows

        Returns
        -------
        pandas dataframe
        """
        fn = glob.glob(data)
        datas = []
        for f in fn:
            datas.append(pd.read_csv(f, sep=sep, skiprows=skiprows))
        return pd.concat(datas, axis=0, ignore_index=True)

    def _columns(self):
        try:
            self.data = self.data[list(self.columns)]
        except KeyError:
            raise Exception("requested columns not in data!")
        if len(self.data.columns) != 4:
            raise Exception("data does not contain required/requested coluumns!")
        if not any(self.data.columns.values == self.columns):
            self.data.columns = self.columns

    def _comp_coding(self):
        if self.comp_coding != ("comp", "incomp"):
            self.data["Comp"] = np.where(
                self.data["Comp"] == self.comp_coding[0], "comp", "incomp"
            )

    def _error_coding(self):
        if self.error_coding != (0, 1):
            self.data["Error"] = np.where(
                self.data["Error"] == self.error_coding[0], 0, 1
            )

    def _outlier(self):
        self.data["outlier"] = np.where(
            (self.data["RT"] <= self.outlier[0]) | (self.data["RT"] >= self.outlier[1]),
            1,
            0,
        )

    def _aggregate_trials(self):
        def aggfun(x):
            new_cols = [
                [
                    len(x["Subject"]),
                    np.sum(x["Error"] == 0),
                    np.sum(x["Error"] == 1),
                    np.sum(x["outlier"] == 1),
                    np.mean(x["RT"][(x["Error"] == 0) & (x["outlier"] == 0)]),
                    np.mean(x["RT"][(x["Error"] == 1) & (x["outlier"] == 0)]),
                ]
            ]
            dat_agg = pd.DataFrame(
                new_cols, columns=["n", "n_cor", "n_err", "n_out", "rt_cor", "rt_err"]
            )
            dat_agg["per_err"] = (
                dat_agg["n_err"] / (dat_agg["n_err"] + dat_agg["n_cor"]) * 100
            )

            return dat_agg

        self.summary_subject = (
            self.data.groupby(["Subject", "Comp"]).apply(aggfun).reset_index()
        ).drop("level_2", axis=1)

    def _aggregate_subjects(self):
        def aggfun(x):
            new_cols = [
                [
                    len(x["Subject"]),
                    np.nanmean(x["rt_cor"]),
                    np.nanstd(x["rt_cor"], ddof=1),
                    np.nanstd(x["rt_cor"], ddof=1) / np.sqrt(x["Subject"].count()),
                    np.nanmean(x["rt_err"]),
                    np.nanstd(x["rt_err"], ddof=1),
                    np.nanstd(x["rt_err"], ddof=1) / np.sqrt(x["Subject"].count()),
                    np.mean(x["per_err"]),
                    np.nanstd(x["per_err"], ddof=1),
                    np.nanstd(x["per_err"], ddof=1) / np.sqrt(x["Subject"].count()),
                ]
            ]
            dat_agg = pd.DataFrame(
                new_cols,
                columns=[
                    "n",
                    "rt_cor",
                    "sd_rt_cor",
                    "se_rt_cor",
                    "rt_err",
                    "sd_rt_err",
                    "se_rt_err",
                    "per_err",
                    "sd_per_err",
                    "se_per_err",
                ],
            )
            return dat_agg

        self.summary = (
            self.summary_subject.groupby(["Comp"]).apply(aggfun).reset_index()
        ).drop("level_1", axis=1)

    def _calc_caf_values(self):
        """Calculate conditional accuracy functions."""

        def caffun(x, n):
            # remove outliers and bin data
            x = x[x.outlier == 0].reset_index()
            cafbin = np.digitize(
                x.loc[:, "RT"],
                np.percentile(x.loc[:, "RT"], np.linspace(0, 100, n + 1)),
            )
            x = x.assign(bin=cafbin)

            return pd.DataFrame((1 - x.groupby(["bin"])["Error"].mean())[:-1])

        self.caf_subject = (
            self.data.groupby(["Subject", "Comp"])
            .apply(caffun, self.n_caf)
            .reset_index()
        )

        def aggfun(x):
            return pd.DataFrame([np.nanmean(x["Error"])], columns=["Error"])

        self.caf = (
            self.caf_subject.groupby(["Comp", "bin"])
            .apply(aggfun)
            .reset_index()
            .drop("level_2", axis=1)
        )

    def _calc_delta_values(self):
        """Calculate compatibility effect + delta values for correct trials."""

        def deltafun(x, n):
            # filter trials
            x = x[(x.outlier == 0) & (x.Error == 0)].reset_index()

            if self.t_delta == 1:

                if self.p_delta is not None:
                    percentiles = self.p_delta
                else:
                    percentiles = np.linspace(0, 1, self.n_delta + 2)[1:-1]

                mean_bins = np.array(
                    [
                        mquantiles(
                            x["RT"][(x["Comp"] == comp)],
                            percentiles,
                            alphap=0.5,
                            betap=0.5,
                        )
                        for comp in ("comp", "incomp")
                    ]
                )

            elif self.t_delta == 2:

                if self.p_delta is not None:
                    percentiles = [0] + self.p_delta + [1]
                else:
                    percentiles = np.linspace(0, 1, self.n_delta + 1)

                mean_bins = np.zeros((2, len(percentiles) - 1))
                for idx, comp in enumerate(("comp", "incomp")):
                    dat = x["RT"][(x["Comp"] == comp)]
                    bin_values = mquantiles(
                        dat,
                        percentiles,
                        alphap=0.5,
                        betap=0.5,
                    )
                    tile = np.digitize(dat, bin_values)
                    mean_bins[idx, :] = np.array(
                        [dat[tile == i].mean() for i in range(1, len(bin_values))]
                    )

            mean_bin = mean_bins.mean(axis=0)
            mean_effect = mean_bins[1, :] - mean_bins[0, :]

            dat = np.array(
                [
                    range(1, len(mean_bin) + 1),
                    mean_bins[0, :],
                    mean_bins[1, :],
                    mean_bin,
                    mean_effect,
                ]
            ).T

            return pd.DataFrame(
                dat,
                columns=["bin", "mean_comp", "mean_incomp", "mean_bin", "mean_effect"],
            )

        self.delta_subject = (
            self.data.groupby(["Subject"]).apply(deltafun, self.n_delta).reset_index()
        ).drop("level_1", axis=1)

        def aggfun(x):
            return pd.DataFrame(
                [
                    [
                        np.nanmean(x["mean_comp"]),
                        np.nanmean(x["mean_incomp"]),
                        np.nanmean(x["mean_bin"]),
                        np.nanmean(x["mean_effect"]),
                    ]
                ],
                columns=["mean_comp", "mean_incomp", "mean_bin", "mean_effect"],
            )

        self.delta = (
            self.delta_subject.groupby(["bin"]).apply(aggfun).reset_index()
        ).drop("level_1", axis=1)

    def plot(self, **kwargs):
        """Plot."""
        DmcPlot(res_th=self, **kwargs).plot()

    def plot(self, **kwargs):
        """Plot."""
        DmcPlot(res_th=self, **kwargs).plot_rt_correct()

    def plot(self, **kwargs):
        """Plot."""
        DmcPlot(res_th=self, **kwargs).plot_er()

    def plot(self, **kwargs):
        """Plot."""
        DmcPlot(res_th=self, **kwargs).plot_rt_error()

    def plot(self, **kwargs):
        """Plot."""
        DmcPlot(res_th=self, **kwargs).plot_cdf()

    def plot(self, **kwargs):
        """Plot."""
        DmcPlot(res_th=self, **kwargs).plot_caf()

    def plot(self, **kwargs):
        """Plot."""
        DmcPlot(res_th=self, **kwargs).plot_delta()
