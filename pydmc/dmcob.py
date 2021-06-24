import pkg_resources
import pandas as pd
from scipy.stats.mstats import mquantiles


def flankerDataRaw():
    """Load Flanker data from Ulrich et al. (2015)."""
    datafile = pkg_resources.resource_stream(__name__, 'data/flankerData.csv')
    return pd.read_csv(datafile, sep="\t")


def simonDataRaw():
    """Load Simon task data from Ulrich et al. (2015)."""
    datafile = pkg_resources.resource_stream(__name__, 'data/simonData.csv')
    return pd.read_csv(datafile, sep="\t")


class DmcOb:
    def __init__(
        self,
        dat,
        n_caf=5,
        n_delta=19,
        outlier=[200, 1200],
        columns=["Subject", "Comp", "RT", "Error"],
        comp_coding=["comp", "incom"],
        error_coding=[0, 1],
        sep="\t",
        skiprows=0,
    ):
        """
        Parameters
        ----------
        """
        self.n_caf = n_caf
        self.n_delta = n_delta
        self.outlier = outlier
        self.columns = columns
        self.comp_coding = comp_coding
        self.error_coding = error_coding

        if not isinstance(dat, pd.DataFrame):
            self.data = self.read_data_files(dat, sep=sep, skiprows=skiprows)
            self._columns()
            self._comp_coding()
            self._error_coding()
            self._outlier()
        else:
            self.data = dat
            self._columns()
            self._comp_coding()
            self._error_coding()
            self._outlier()

        self._aggregate_trials()
        self._aggregate_subjects()
        self._calc_caf_values()
        self._calc_delta_values()

    @staticmethod
    def read_data_files(dat, sep="\t", skiprows=0):
        fn = glob.glob(dat)
        dats = []
        for f in fn:
            dats.append(pd.read_csv(f, sep=sep, skiprows=skiprows))
        return pd.concat(dats, axis=0, ignore_index=True)

    def _columns(self):
        try:
            self.data = self.data[self.columns]
        except:
            raise Exception("requested columns not in data!")
        if len(self.data.columns) != 4:
            raise Exception("data does not contain required/requested coluumns!")
        if not any(self.data.columns.values == self.columns):
            self.data.columns = self.columns

    def _comp_coding(self):
        if self.comp_coding != ["comp", "incomp"]:
            self.data["Comp"] = np.where(
                self.data["Comp"] == self.comp_coding[0], "comp", "incomp"
            )

    def _error_coding(self):
        if self.error_coding != [0, 1]:
            self.data["Error"] = np.where(
                self.data["Error"] == self.error_coding[0], 0, 1
            )

    def _outlier(self):
        self.data["outlier"] = np.where(
            (self.data["RT"] <= self.outlier[0]) | (self.data["RT"] >= self.outlier[1]),
            True,
            False,
        )

    def _aggregate_trials(self):
        def aggfun(x):
            new_cols = [
                [
                    len(x["Subject"]),
                    np.sum(x["Error"] == False),
                    np.sum(x["Error"] == True),
                    np.sum(x["outlier"] == True),
                    np.mean(x["RT"][(x["Error"] == False) & (x["outlier"] == False)]),
                    np.mean(x["RT"][(x["Error"] == True) & (x["outlier"] == False)]),
                ]
            ]
            dat_agg = pd.DataFrame(
                new_cols, columns=["N", "nCor", "nErr", "nOut", "rtCor", "rtErr"]
            )
            dat_agg["perErr"] = (
                dat_agg["nErr"] / (dat_agg["nErr"] + dat_agg["nCor"]) * 100
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
                    np.nanmean(x["rtCor"]),
                    np.nanstd(x["rtCor"], ddof=1),
                    np.nanstd(x["rtCor"], ddof=1) / np.sqrt(x["Subject"].count()),
                    np.nanmean(x["rtErr"]),
                    np.nanstd(x["rtErr"], ddof=1),
                    np.nanstd(x["rtErr"], ddof=1) / np.sqrt(x["Subject"].count()),
                    np.mean(x["perErr"]),
                    np.nanstd(x["perErr"], ddof=1),
                    np.nanstd(x["perErr"], ddof=1) / np.sqrt(x["Subject"].count()),
                ]
            ]
            dat_agg = pd.DataFrame(
                new_cols,
                columns=[
                    "N",
                    "rtCor",
                    "sdRtCor",
                    "seRtCor",
                    "rtErr",
                    "sdRtErr",
                    "seRtErr",
                    "perErr",
                    "sdPerErr",
                    "sePerErr",
                ],
            )
            return dat_agg

        self.summary = (
            self.summary_subject.groupby(["Comp"]).apply(aggfun).reset_index()
        ).drop("level_1", axis=1)

    def _calc_caf_values(self):
        """Calculate conditional accuracy functions."""

        def caffun(x, n):
            # remove outliers
            x = x[x.outlier == False].reset_index()
            # bin data
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
            x = x[(x.outlier == False) & (x.Error == 0)].reset_index()

            nbin = np.arange(1, n + 1)
            mean_comp = mquantiles(
                x["RT"][(x["Comp"] == "comp")],
                np.linspace(0, 1, n + 2)[1:-1],
                alphap=0.5,
                betap=0.5,
            )

            mean_incomp = mquantiles(
                x["RT"][(x["Comp"] == "incomp")],
                np.linspace(0, 1, n + 2)[1:-1],
                alphap=0.5,
                betap=0.5,
            )
            mean_bin = (mean_comp + mean_incomp) / 2
            mean_effect = mean_incomp - mean_comp

            dat = np.array([nbin, mean_comp, mean_incomp, mean_bin, mean_effect]).T

            return pd.DataFrame(
                dat,
                columns=["Bin", "mean_comp", "mean_incomp", "mean_bin", "mean_effect"],
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
            self.delta_subject.groupby(["Bin"]).apply(aggfun).reset_index()
        ).drop("level_1", axis=1)

    def plot(self):
        """Plot."""

        # upper left panel (rt correct)
        plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=1)
        self.plot_rt_correct(show=False)

        # middle left pannel
        plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=1)
        self.plot_er(show=False)

        # bottom left pannel
        plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=1)
        self.plot_rt_error(show=False)

        # upper right panel (cdf)
        plt.subplot2grid((3, 2), (0, 1), rowspan=1, colspan=1)
        self.plot_cdf(show=False)

        # middle right (left) panel (PDF)
        plt.subplot2grid((3, 2), (1, 1), rowspan=1, colspan=1)
        self.plot_caf(show=False)

        # lower right (right) panel (CDF)
        plt.subplot2grid((3, 2), (2, 1), rowspan=1, colspan=1)
        self.plot_delta(show=False)

        plt.show(block=False)

    def plot_rt_correct(
        self,
        show=True,
        ylim=None,
        xlabel=None,
        cond_labels=["Compatible", "Incompatible"],
        ylabel="RT Correct [ms]",
    ):
        """Plot correct RT's."""

        plt.plot(cond_labels, self.summary["rtCor"], "ko-")

        if ylim is None:
            ylim = [
                np.min(self.summary["rtCor"]) - 100,
                np.max(self.summary["rtCor"]) + 100,
            ]

        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.margins(x=0.4)

        if show:
            plt.show(block=False)

    def plot_er(
        self,
        show=True,
        ylim=None,
        xlabel=None,
        cond_labels=["Compatible", "Incompatible"],
        ylabel="Error Rate [%]",
    ):
        """Plot error rate"""

        plt.plot(cond_labels, self.summary["perErr"], "ko-")

        if ylim is None:
            ylim = [0, np.max(self.summary["perErr"]) + 5]

        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.margins(x=0.4)

        if show:
            plt.show(block=False)

    def plot_rt_error(
        self,
        show=True,
        ylim=None,
        xlabel=None,
        cond_labels=["Compatible", "Incompatible"],
        ylabel="RT Error [ms]",
    ):
        """Plot error RT's."""

        plt.plot(cond_labels, self.summary["rtErr"], "ko-")

        if ylim is None:
            ylim = [
                np.min(self.summary["rtErr"]) - 100,
                np.max(self.summary["rtErr"]) + 100,
            ]

        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.margins(x=0.4)

        if show:
            plt.show(block=False)

    def plot_cdf(
        self,
        show=True,
        xlim=None,
        ylim=[0, 1.05],
        xlabel=None,
        cond_labels=["Compatible", "Incompatible"],
        ylabel="RT Error [ms]",
        cols=("green", "red"),
    ):
        """Plot CDF"""
        plt.plot(
            self.delta["mean_comp"],
            np.linspace(0, 1, self.n_delta + 2)[1:-1],
            color=cols[0],
            linestyle="-",
            marker="o",
        )
        plt.plot(
            self.delta["mean_incomp"],
            np.linspace(0, 1, self.n_delta + 2)[1:-1],
            color=cols[1],
            linestyle="-",
            marker="o",
        )

        if xlim is None:
            xlim = [
                np.min(self.delta.mean_bin) - 100,
                np.max(self.delta.mean_bin) + 100,
            ]

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if show:
            plt.show(block=False)

    def plot_caf(
        self,
        show=True,
        xlabel="RT Bin",
        ylabel="CAF",
        ylim=[0, 1.1],
        cols=("green", "red"),
    ):
        """Plot CAF."""
        plt.plot(
            self.caf["Error"][self.caf["Comp"] == "comp"],
            color=cols[0],
            linestyle="-",
            marker="o",
        )
        plt.plot(
            self.caf["Error"][self.caf["Comp"] == "incomp"].reset_index(),
            color=cols[1],
            linestyle="-",
            marker="o",
        )

        plt.ylim(ylim)
        plt.xticks(range(0, self.n_caf), [str(x) for x in range(1, self.n_caf + 1)])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if show:
            plt.show(block=False)

    def plot_delta(
        self, show=True, xlabel="Time (ms)", ylabel=r"$\Delta$", xlim=None, ylim=None
    ):
        """Plot reaction-time delta plots."""

        plt.plot(self.delta["mean_bin"], self.delta["mean_effect"], "ko-", markersize=4)

        if xlim is None:
            xlim = [
                np.min(self.delta.mean_bin) - 100,
                np.max(self.delta.mean_bin) + 100,
            ]
        if ylim is None:
            ylim = [
                np.min(self.delta.mean_effect) - 25,
                np.max(self.delta.mean_effect) + 25,
            ]

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if show:
            plt.show(block=False)


if __name__ == "__main__":
    dmc = DmcOb()
