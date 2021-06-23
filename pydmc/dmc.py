"""
DMC model simulation detailed in  Ulrich, R., Schröter, H., Leuthold, H.,
& Birngruber, T. (2015). Automatic and controlled stimulus processing
in conflict tasks: Superimposed diffusion processes and delta functions.
Cognitive Psychology, 78, 148-174.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from numba import jit, prange
from fastkde import fastKDE


class DmcSim:
    def __init__(
        self,
        amp=20,
        tau=30,
        aa_shape=2,
        drc=0.5,
        sigma=4,
        bnds=75,
        res_mean=300,
        res_sd=30,
        n_trls=100000,
        t_max=1000,
        var_sp=False,
        sp_shape=3,
        var_dr=False,
        dr_lim=(0.1, 0.7),
        dr_shape=3,
        n_caf=5,
        n_delta=19,
        full_data=False,
        n_trls_data=5,
        run_simulation=True,
        plt_figs=False,
    ):
        """
        Parameters
        ----------
        amp: int/float, optional
            amplitude of automatic activation
        tau: int/float, optional
            time to peak automatic activation
        aa_shape: int, optional
            shape parameter of automatic activation
        drc: int/float, optional
            drift rate of controlled processes
        sigma: int, optional
            diffusion constant
        bnds: int, optional
            +- response barrier
        res_mean: int/float, optional
            mean of non-decisional component
        res_sd: int/float, optional
            standard deviation of non-decisional component
        n_trls: int (1000 to 100000 (default)), optional
            number of trials
        t_max: int, optional
            number of time points per trial
        var_sp: bool, optional
            variable starting point
        sp_shape: int, optional
            shape parameter of starting point distribution
        var_dr: bool, optional
            variable drift rate
        dr_lim: tuple, optional
            limit range of distribution of drift rate
        dr_shape: int, optional
            shape parameter of drift rate
        plt_figs: bool, optional
            plot figures
        n_caf: range, optional
            caf bins
        n_delta: range, optional
            delta reaction time bins
        full_data: bool, optional
            run simulation to t_max to caluculate activation
        n_trls_data: int, optional
            number of individual trials to store
        run_simulation=True, optional
            run simulation
        plt_figs=False, optional
            plot data

        Returns
        -------
        - DmcSim object

        Notes
        -----
        - Tested with Python 3.9

        Examples
        --------
        >>> from pydmc.dmc import DmcSim
        >>> dmc = DmcSim(full_data=True)
        >>> dmc.plot()                 # Fig 3
        >>> dat = DmcSim()
        >>> dmc.plot()                 # Fig 3 (part)
        >>> dat = DmcSim(tau = 150)
        >>> dmc.plot()                 # Fig 4
        >>> dat = DmcSim(tau = 90)
        >>> dmc.plot()                 # Fig 5
        >>> dat = DmcSim(var_sp = True)
        >>> dmc.plot()                 # Fig 6
        >>> dat = DmcSim(var_dr = True)
        >>> dmc.plot()                 # Fig 7
        """

        self.amp = amp
        self.tau = tau
        self.aa_shape = aa_shape
        self.drc = drc
        self.sigma = sigma
        self.bnds = bnds
        self.res_mean = res_mean
        self.res_sd = res_sd
        self.n_trls = n_trls
        self.t_max = t_max
        self.var_dr = var_dr
        self.dr_lim = dr_lim
        self.dr_shape = dr_shape
        self.var_sp = var_sp
        self.sp_shape = sp_shape
        self.n_caf = n_caf
        self.n_delta = n_delta
        self.full_data = full_data
        self.n_trls_data = n_trls_data

        self.tim = np.arange(1, self.t_max + 1, 1)
        self.eq4 = (
            self.amp
            * np.exp(-self.tim / self.tau)
            * (np.exp(1) * self.tim / (self.aa_shape - 1) / self.tau)
            ** (self.aa_shape - 1)
        )
        self.dat = []
        self.dat_trials = []
        self.xt = []
        self.summary = []
        self.caf = []
        self.delta = []

        if run_simulation:
            self.run_simulation()

        if plt_figs:
            self.plot()

    def run_simulation(self):
        """Run simulation."""

        if self.full_data:
            self._run_simulation_numpy()
        else:
            self._run_simulation_numba()

    def _run_simulation_numpy(self):
        """Run simulation using numpy."""

        rand_nums = None
        for comp in [1, -1]:
            if rand_nums is None:
                rand_nums = np.random.randn(self.n_trls, self.t_max)
            else:
                np.random.shuffle(rand_nums)

            dr = self._dr()
            sp = self._sp()
            drc = (
                comp * self.eq4 * ((self.aa_shape - 1) / self.tim - 1 / self.tau)
                + np.tile(dr, (self.t_max, 1)).T
            )

            # random process
            xt = drc + (self.sigma * rand_nums)

            # variable starting point
            xt[:, 0] += sp

            # cumulate activation over time
            xt = np.cumsum(xt, 1)

            # reaction time
            rt = np.argmax(np.abs(xt) > self.bnds, axis=1) + 1
            rt[rt == 1] = self.t_max

            self.dat.append(
                np.vstack(
                    (
                        rt + np.random.normal(self.res_mean, self.res_sd, self.n_trls),
                        xt[np.arange(len(xt)), rt - 1] < self.bnds,
                    )
                )
            )
            self.dat_trials.append(xt[0 : self.n_trls_data])
            self.xt.append(xt.mean(0))

        self._calc_caf_values()
        self._calc_delta_values()
        self._results_summary()

    def _run_simulation_numba(self):
        """Run simulation using numba."""

        for comp in [1, -1]:

            dr = self._dr()
            sp = self._sp()
            drc = comp * self.eq4 * ((self.aa_shape - 1) / self.tim - 1 / self.tau)

            self.dat.append(
                _run_simulation_numba(
                    drc,
                    sp,
                    dr,
                    self.t_max,
                    self.sigma,
                    self.res_mean,
                    self.res_sd,
                    self.bnds,
                    self.n_trls,
                )
            )

        self._calc_caf_values()
        self._calc_delta_values()
        self._results_summary()

    def _results_summary(self):
        """Create results summary table."""

        summary = [
            [
                round(np.mean(self.dat[0][0][self.dat[0][1] == 0])),
                round(np.std(self.dat[0][0][self.dat[0][1] == 0])),
                round(np.sum(self.dat[0][1] / self.n_trls) * 100, 1),
                round(np.mean(self.dat[0][0][self.dat[0][1] == 1])),
                round(np.std(self.dat[0][0][self.dat[0][1] == 1])),
            ],
            [
                round(np.mean(self.dat[1][0][self.dat[1][1] == 0])),
                round(np.std(self.dat[1][0][self.dat[1][1] == 0])),
                round(np.sum(self.dat[1][1] / self.n_trls) * 100, 1),
                round(np.mean(self.dat[1][0][self.dat[1][1] == 1])),
                round(np.std(self.dat[1][0][self.dat[1][1] == 1])),
            ],
        ]

        self.summary = pd.DataFrame(
            summary,
            index=["comp", "incomp"],
            columns=["rtCorr", "sdCorr", "perErr", "rtErr", "sdRtErr"],
        )

    def _calc_caf_values(self):
        """Calculate conditional accuracy functions."""

        def caffun(x, n):
            # bin data
            bin = np.digitize(
                x.loc[:, "RT"],
                np.percentile(x.loc[:, "RT"], np.linspace(0, 100, self.n_caf + 1)),
            )
            x = x.assign(bin=bin)

            return pd.DataFrame((1 - x.groupby(["bin"])["Error"].mean())[:-1])

        # create temp pandas dataframe
        dfc = pd.DataFrame(self.dat[0].T, columns=["RT", "Error"]).assign(Comp="comp")
        dfi = pd.DataFrame(self.dat[1].T, columns=["RT", "Error"]).assign(Comp="incomp")
        df = pd.concat([dfc, dfi])

        self.caf = df.groupby(["Comp"]).apply(caffun, self.n_caf).reset_index()

    def _calc_delta_values(self):
        """Calculate compatibility effect + delta values for correct trials."""

        nbin = np.arange(1, self.n_delta + 1)
        mean_comp = np.percentile(
            self.dat[0][0], np.linspace(0, 100, self.n_delta + 2)[1:-1]
        )
        mean_incomp = np.percentile(
            self.dat[1][0], np.linspace(0, 100, self.n_delta + 2)[1:-1]
        )
        mean_bin = (mean_comp + mean_incomp) / 2
        mean_effect = mean_incomp - mean_comp

        dat = np.array([nbin, mean_comp, mean_incomp, mean_bin, mean_effect]).T

        self.delta = pd.DataFrame(
            dat, columns=["Bin", "mean_comp", "mean_incom", "mean_bin", "mean_effect"]
        )

    @staticmethod
    def rand_beta(lim=(0, 1), shape=3, n_trls=1):
        """Return random vector between limits weighted by beta function."""
        x = np.random.beta(shape, shape, n_trls)
        x = x * (lim[1] - lim[0]) + lim[0]

        return x

    def plot(self):
        """Plot"""
        if self.full_data:
            self._plot_full()
        else:
            self._plot()

    def _plot_full(self):

        # upper left panel (activation)
        plt.subplot2grid((6, 4), (0, 0), rowspan=3, colspan=2)
        self.plot_activation(show=False)

        # lower left panel (trials)
        plt.subplot2grid((6, 4), (3, 0), rowspan=3, colspan=2)
        self.plot_trials(show=False)

        # upper right (left) panel (PDF)
        plt.subplot2grid((6, 4), (0, 2), rowspan=2)
        self.plot_pdf(show=False)

        # upper right (right) panel (CDF)
        plt.subplot2grid((6, 4), (0, 3), rowspan=2)
        self.plot_cdf(show=False)

        # middle right panel (CAF)
        plt.subplot2grid((6, 4), (2, 2), rowspan=2, colspan=2)
        self.plot_caf(show=False)

        # bottom right panel (delta)
        plt.subplot2grid((6, 4), (4, 2), rowspan=2, colspan=2)
        self.plot_delta(show=False)

        plt.subplots_adjust(hspace=1.5, wspace=0.35)
        plt.show(block=False)

    def _plot(self):

        # upper right (left) panel (PDF)
        plt.subplot2grid((3, 2), (0, 0), rowspan=1)
        self.plot_pdf(show=False)

        # upper right (eight) panel (CDF)
        plt.subplot2grid((3, 2), (0, 1), rowspan=1)
        self.plot_cdf(show=False)

        # middle left panel
        plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=2)
        self.plot_caf(show=False)

        # bottom right panel
        plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=2)
        self.plot_delta(show=False)

        plt.subplots_adjust(hspace=1.5, wspace=0.35)
        plt.show(block=False)

    def plot_activation(
        self,
        show=True,
        xlim=None,
        ylim=None,
        xlabel="Time (ms)",
        ylabel="E[X(t)]",
    ):
        """Plot activation."""

        if not self.xt:
            print("Plotting activation function requires full_data=True")
            return

        plt.plot(self.eq4, "k")
        plt.plot(self.eq4 * -1, "k--")
        plt.plot(self.xt[0], "g")
        plt.plot(self.xt[1], "r")
        plt.plot(np.cumsum(np.repeat(self.drc, self.t_max)), "k")

        if xlim is None:
            xlim = [0, self.t_max]

        if ylim is None:
            ylim = [-self.bnds - 20, self.bnds + 20]

        plt.plot(xlim, [self.bnds, self.bnds], "k--")
        plt.plot(xlim, [-self.bnds, -self.bnds], "k--")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if show:
            plt.show(block=False)

    def plot_trials(
        self,
        show=True,
        xlabel="Time (ms)",
        ylabel="X(t)",
        xlim=None,
        ylim=None,
        cols=("green", "red"),
    ):
        """Plot individual trials."""

        if not self.xt:
            print("Plotting individual trials function requires full_data=True")
            return

        for trl in range(5):
            idx = np.where(np.abs(self.dat_trials[0][trl, :]) >= self.bnds)[0][0]
            plt.plot(self.dat_trials[0][trl][0:idx], cols[0])
            idx = np.where(np.abs(self.dat_trials[1][trl, :]) >= self.bnds)[0][0]
            plt.plot(self.dat_trials[1][trl][0:idx], cols[1])

        if xlim is None:
            xlim = [0, self.t_max]

        if ylim is None:
            ylim = [-self.bnds - 20, self.bnds + 20]

        plt.plot(xlim, [self.bnds, self.bnds], "k--")
        plt.plot(xlim, [-self.bnds, -self.bnds], "k--")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if show:
            plt.show(block=False)

    def plot_pdf(
        self,
        show=True,
        xlim=None,
        ylim=None,
        xlabel="Time (ms)",
        ylabel="PDF",
        cols=("green", "red"),
    ):
        """Plot PDF."""

        comp_pdf, axes1 = fastKDE.pdf(self.dat[0][0])
        incomp_pdf, axes2 = fastKDE.pdf(self.dat[1][0])

        plt.plot(axes1, comp_pdf, cols[0])
        plt.plot(axes2, incomp_pdf, cols[1])

        if xlim is None:
            xlim = [0, self.t_max]

        if ylim is None:
            ylim = [0, 0.01]

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if show:
            plt.show(block=False)

    def plot_cdf(
        self,
        show=True,
        xlabel="Time (ms)",
        ylabel="CDF",
        xlim=None,
        ylim=[0, 1.0],
        cols=("green", "red"),
    ):
        """Plot CDF."""

        comp_pdf, axes1 = fastKDE.pdf(self.dat[0][0])
        incomp_pdf, axes2 = fastKDE.pdf(self.dat[1][0])

        plt.plot(axes1, np.cumsum(comp_pdf) * np.diff(axes1)[0:1], cols[0])
        plt.plot(axes2, np.cumsum(incomp_pdf) * np.diff(axes2)[0:1], cols[1])

        if xlim is None:
            xlim = [0, self.t_max]

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

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
            cols[0],
            linestyle="-",
            marker="o",
        )
        plt.plot(
            self.caf["Error"][self.caf["Comp"] == "incomp"].reset_index(),
            cols[1],
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
            xlim = [0, self.t_max]
        if ylim is None:
            ylim = [-50, 100]

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if show:
            plt.show(block=False)

    def _dr(self):
        if self.var_dr:
            return self.rand_beta(self.dr_lim, self.dr_shape, self.n_trls)
        else:
            return np.ones(self.n_trls) * self.drc

    def _sp(self):
        if self.var_sp:
            return self.rand_beta((-self.bnds, self.bnds), self.sp_shape, self.n_trls)
        else:
            return np.zeros(self.n_trls)


@jit(nopython=True, parallel=True)
def _run_simulation_numba(drc, sp, dr, t_max, sigma, res_mean, res_sd, bnds, n_trls):

    dat = np.vstack((np.ones(n_trls) * t_max, np.zeros(n_trls)))

    for trl in prange(n_trls):
        trl_xt = sp[trl]
        for tp in range(0, t_max):
            trl_xt += drc[tp] + dr[trl] + (sigma * np.random.randn())
            if np.abs(trl_xt) >= bnds:
                dat[0, trl] = tp + np.random.normal(res_mean, res_sd)
                dat[1, trl] = trl_xt < 0.0
                break

    return dat


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
            new_cols = {
                "N": len(x["Subject"]),
                "nCor": np.sum(x["Error"] == False),
                "nErr": np.sum(x["Error"] == True),
                "nOut": np.sum(x["outlier"] == True),
                "rtCor": np.mean(
                    x["RT"][(x["Error"] == False) & (x["outlier"] == False)]
                ),
                "rtErr": np.mean(
                    x["RT"][(x["Error"] == True) & (x["outlier"] == False)]
                ),
            }
            dat_agg = pd.Series(
                new_cols, index=["N", "nCor", "nErr", "nOut", "rtCor", "rtErr"]
            )
            dat_agg["perErr"] = (
                dat_agg["nErr"] / (dat_agg["nErr"] + dat_agg["nCor"]) * 100
            )

            return dat_agg

        self.summary_subject = (
            self.data.groupby(["Subject", "Comp"]).apply(aggfun).reset_index()
        )

    def _aggregate_subjects(self):
        def aggfun(x):
            new_cols = {
                "N": len(x["Subject"]),
                "rtCor": np.nanmean(x["rtCor"]),
                "sdRtCor": np.nanstd(x["rtCor"], ddof=1),
                "seRtCor": np.nanstd(x["rtCor"], ddof=1)
                / np.sqrt(x["Subject"].count()),
                "rtErr": np.nanmean(x["rtErr"]),
                "sdRtErr": np.nanstd(x["rtErr"], ddof=1),
                "seRtErr": np.nanstd(x["rtErr"], ddof=1)
                / np.sqrt(x["Subject"].count()),
                "perErr": np.mean(x["perErr"]),
                "sdPerErr": np.nanstd(x["perErr"], ddof=1),
                "sePerErr": np.nanstd(x["perErr"], ddof=1)
                / np.sqrt(x["Subject"].count()),
            }
            dat_agg = pd.Series(
                new_cols,
                index=[
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
        )

    def _calc_caf_values(self):
        """Calculate conditional accuracy functions."""

        def caffun(x, n):
            # remove outliers
            x = x[x.outlier == False].reset_index()
            # bin data
            bin = np.digitize(
                x.loc[:, "RT"],
                np.percentile(x.loc[:, "RT"], np.linspace(0, 100, n + 1)),
            )
            x = x.assign(bin=bin)

            return pd.DataFrame((1 - x.groupby(["bin"])["Error"].mean())[:-1])

        self.caf_subject = (
            self.data.groupby(["Subject", "Comp"])
            .apply(caffun, self.n_caf)
            .reset_index()
        )

    def plot(self):
        """Plot"""

        # upper left panel (rt correct)
        plt.subplot2grid((3, 2), (0, 0), rowspan=3, colspan=2)
        self.plot_rt_correct(show=False)

        # # middle left pannel
        # plt.subplot2grid((3, 2), (0, 1), rowspan=3, colspan=2)
        # self.plot_er(show=False)

        # # bottom left pannel
        # plt.subplot2grid((3, 2), (0, 2), rowspan=3, colspan=2)
        # self.plot_rt_error(show=False)

        # # upper right panel (cdf)
        # plt.subplot2grid((3, 2), (1, 0), rowspan=3, colspan=2)
        # self.plot_cdf(show=False)

        # # middle right (left) panel (PDF)
        # plt.subplot2grid((3, 2), (1, 1), rowspan=3, colspan=2)
        # self.plot_pdf(show=False)

        # # lower right (right) panel (CDF)
        # plt.subplot2grid((3, 2), (1, 2), rowspan=3, colspan=2)
        # self.plot_cdf(show=False)

        plt.show(block=False)

    def plot_rt_correct(
        self,
        show=True,
        xlim=None,
        ylim=None,
        xlabel=None,
        cond_labels=["Compatible", "Incompatible"],
        ylabel="RT Correct [ms]",
    ):
        """Plot correct RT's."""

        plt.plot(cond_labels, self.summary["rtCor"], "ko-")

        if ylim is None:
            ylim = [np.min(self.summary["rtCor"])-100, np.max(self.summary["rtCor"])+100]

        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.margins(x=0.4)

        if show:
            plt.show(block=False)


if __name__ == "__main__":
    dmc = DmcSim()
