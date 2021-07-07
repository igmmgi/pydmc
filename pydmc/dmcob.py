import glob
import pkg_resources
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles


def flanker_data_raw():
    """Load Flanker data from Ulrich et al. (2015)."""
    datafile = pkg_resources.resource_stream(__name__, "data/flankerData.csv")
    return pd.read_csv(datafile, sep="\t")


def simon_data_raw():
    """Load Simon task data from Ulrich et al. (2015)."""
    datafile = pkg_resources.resource_stream(__name__, "data/simonData.csv")
    return pd.read_csv(datafile, sep="\t")


class DmcOb:
    def __init__(
        self,
        data,
        n_caf=5,
        n_delta=19,
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
        """
        self.n_caf = n_caf
        self.n_delta = n_delta
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
            x = x[(x.outlier == 0) & (x.Error == 0)].reset_index()

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

    def plot(
        self, label_fontsize=12, tick_fontsize=10, hspace=0.5, wspace=0.5, **kwargs
    ):
        """Plot."""

        # upper left panel (rt correct)
        plt.subplot2grid((3, 2), (0, 0))
        self.plot_rt_correct(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # middle left pannel
        plt.subplot2grid((3, 2), (1, 0))
        self.plot_er(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # bottom left pannel
        plt.subplot2grid((3, 2), (2, 0))
        self.plot_rt_error(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # upper right panel (cdf)
        plt.subplot2grid((3, 2), (0, 1))
        self.plot_cdf(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # middle right (left) panel (PDF)
        plt.subplot2grid((3, 2), (1, 1))
        self.plot_caf(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # lower right (right) panel (CDF)
        plt.subplot2grid((3, 2), (2, 1))
        self.plot_delta(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        plt.show(block=False)

    def plot_rt_correct(
        self,
        show=True,
        ylim=None,
        xlabel=None,
        cond_labels=("Compatible", "Incompatible"),
        ylabel="RT Correct [ms]",
        label_fontsize=12,
        tick_fontsize=10,
        **kwargs
    ):
        """Plot correct RT's."""

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        plt.plot(cond_labels, self.summary["rt_cor"], **kwargs)

        ylim = ylim or [np.min(self.summary["rt_cor"]) - 100, np.max(self.summary["rt_cor"]) + 100]

        plt.margins(x=0.5)
        _adjust_plt(None, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if show:
            plt.show(block=False)

    def plot_er(
        self,
        show=True,
        ylim=None,
        xlabel=None,
        cond_labels=("Compatible", "Incompatible"),
        ylabel="Error Rate [%]",
        label_fontsize=12,
        tick_fontsize=10,
        **kwargs
    ):
        """Plot error rate."""

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        plt.plot(cond_labels, self.summary["per_err"], **kwargs)

        ylim = ylim or [0, np.max(self.summary["per_err"]) + 5]

        plt.margins(x=0.5)
        _adjust_plt(None, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if show:
            plt.show(block=False)

    def plot_rt_error(
        self,
        show=True,
        ylim=None,
        xlabel=None,
        cond_labels=("Compatible", "Incompatible"),
        ylabel="RT Error [ms]",
        label_fontsize=12,
        tick_fontsize=10,
        **kwargs
    ):
        """Plot error RT's."""

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        plt.plot(cond_labels, self.summary["rt_err"], **kwargs)

        ylim = ylim or [np.min(self.summary["rt_err"]) - 100, np.max(self.summary["rt_err"]) + 100]

        plt.margins(x=0.5)
        _adjust_plt(None, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if show:
            plt.show(block=False)

    def plot_cdf(
        self,
        show=True,
        xlim=None,
        ylim=(0, 1.05),
        xlabel=None,
        cond_labels=("Compatible", "Incompatible"),
        ylabel="CDF",
        label_fontsize=12,
        tick_fontsize=10,
        legend_position="lower right",
        colors=("green", "red"),
        **kwargs
    ):
        """Plot CDF."""

        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        plt.plot(
            self.delta["mean_comp"],
            np.linspace(0, 1, self.n_delta + 2)[1:-1],
            color=colors[0],
            label=cond_labels[0],
            **kwargs,
        )
        plt.plot(
            self.delta["mean_incomp"],
            np.linspace(0, 1, self.n_delta + 2)[1:-1],
            color=colors[1],
            label=cond_labels[1],
            **kwargs,
        )

        xlim = xlim or [np.min(self.delta.mean_bin) - 100, np.max(self.delta.mean_bin) + 100]
        ylim = ylim or [0, 1.05]

        _adjust_plt(xlim, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def plot_caf(
        self,
        show=True,
        ylim=(0, 1.1),
        xlabel="RT Bin",
        ylabel="CAF",
        label_fontsize=12,
        tick_fontsize=10,
        cond_labels=("Compatible", "Incompatible"),
        legend_position="lower right",
        colors=("green", "red"),
        **kwargs
    ):
        """Plot CAF."""

        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        plt.plot(
            self.caf["bin"][self.caf["Comp"] == "comp"],
            self.caf["Error"][self.caf["Comp"] == "comp"],
            color=colors[0],
            label=cond_labels[0],
            **kwargs,
        )

        plt.plot(
            self.caf["bin"][self.caf["Comp"] == "incomp"],
            self.caf["Error"][self.caf["Comp"] == "incomp"],
            color=colors[1],
            label=cond_labels[1],
            **kwargs,
        )

        plt.xticks(range(1, self.n_caf + 1), [str(x) for x in range(1, self.n_caf + 1)])

        _adjust_plt(None, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def plot_delta(
        self,
        show=True,
        xlabel="Time (ms)",
        ylabel=r"$\Delta$",
        label_fontsize=12,
        tick_fontsize=10,
        xlim=None,
        ylim=None,
        **kwargs
    ):
        """Plot reaction-time delta plots."""

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        plt.plot(self.delta["mean_bin"], self.delta["mean_effect"], **kwargs)

        xlim = xlim or [np.min(self.delta.mean_bin) - 100, np.max(self.delta.mean_bin) + 100]
        ylim = ylim or [np.min(self.delta.mean_effect) - 25, np.max(self.delta.mean_effect) + 25]

        plt.margins(x=0.5)
        _adjust_plt(xlim, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if show:
            plt.show(block=False)


def _adjust_plt(xlim, ylim, xlabel, ylabel, label_fontsize, tick_fontsize):
    """Internal function to adjust some plot properties."""
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.yticks(fontsize=tick_fontsize)
