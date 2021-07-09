"""
DMC model simulation detailed in  Ulrich, R., Schröter, H., Leuthold, H.,
& Birngruber, T. (2015). Automatic and controlled stimulus processing
in conflict tasks: Superimposed diffusion processes and delta functions.
Cognitive Psychology, 78, 148-174.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit, prange
from fastkde import fastKDE
from scipy.stats.mstats import mquantiles


class DmcSim:
    """DMC Simulation."""

    def __init__(
        self,
        amp=20,
        tau=30,
        aa_shape=2,
        drc=0.5,
        sigma=4,
        bnds=75,
        res_dist=1,
        res_mean=300,
        res_sd=30,
        n_trls=100000,
        t_max=1000,
        var_sp=False,
        sp_lim=(-75, 75),
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
        res_dist: int, optional
            non-decisional component distribution (1=normal, 2=uniform)
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
        sp_lim: tuple, optional
            limiit range of distribution of starting point
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
        >>> from pydmc.dmcsim import DmcSim
        >>> dmc_sim = DmcSim(full_data=True)
        >>> dmc_sim.plot()                 # Fig 3
        >>> dmc_sim = DmcSim()
        >>> dmc_sim.plot()                 # Fig 3 (part)
        >>> dmc_sim = DmcSim(tau = 150)
        >>> dmc_sim.plot()                 # Fig 4
        >>> dmc_sim = DmcSim(tau = 90)
        >>> dmc_sim.plot()                 # Fig 5
        >>> dmc_sim = DmcSim(var_sp = True)
        >>> dmc_sim.plot()                 # Fig 6
        >>> dmc_sim = DmcSim(var_dr = True)
        >>> dmc_sim.plot()                 # Fig 7
        """

        self.amp = amp
        self.tau = tau
        self.aa_shape = aa_shape
        self.drc = drc
        self.sigma = sigma
        self.bnds = bnds
        self.res_dist = res_dist
        self.res_mean = res_mean
        self.res_sd = res_sd
        self.n_trls = n_trls
        self.t_max = t_max
        self.var_dr = var_dr
        self.dr_lim = dr_lim
        self.dr_shape = dr_shape
        self.var_sp = var_sp
        self.sp_lim = sp_lim
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
            self._run_simulation_full()
        else:
            self._run_simulation()

        self._calc_caf_values()
        self._calc_delta_values()
        self._results_summary()

    def _run_simulation(self):
        """Run simulation using numba."""

        for comp in [1, -1]:
            dr = self._dr()
            sp = self._sp()
            drc = comp * self.eq4 * ((self.aa_shape - 1) / self.tim - 1 / self.tau)

            self.dat.append(
                _run_simulation(
                    drc,
                    sp,
                    dr,
                    self.t_max,
                    self.sigma,
                    self.res_dist,
                    self.res_mean,
                    self.res_sd,
                    self.bnds,
                    self.n_trls,
                )
            )

    def _run_simulation_full(self):
        """Run simulation using numba."""

        for comp in [1, -1]:
            dr = self._dr()
            sp = self._sp()
            drc = comp * self.eq4 * ((self.aa_shape - 1) / self.tim - 1 / self.tau)

            activation, trials, dat = _run_simulation_full(
                drc,
                sp,
                dr,
                self.t_max,
                self.sigma,
                self.res_dist,
                self.res_mean,
                self.res_sd,
                self.bnds,
                self.n_trls,
                self.n_trls_data,
            )

            self.xt.append(activation)
            self.dat_trials.append(trials)
            self.dat.append(dat)

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
            columns=["rt_cor", "sd_cor", "per_err", "rt_err", "sd_rt_err"],
        )

    def _calc_caf_values(self):
        """Calculate conditional accuracy functions."""

        def caffun(x, n):
            cafbin = np.digitize(
                x.loc[:, "RT"],
                np.percentile(x.loc[:, "RT"], np.linspace(0, 100, n + 1)),
            )
            x = x.assign(bin=cafbin)

            return pd.DataFrame((1 - x.groupby(["bin"])["Error"].mean())[:-1])

        # create temp pandas dataframe
        dfc = pd.DataFrame(self.dat[0].T, columns=["RT", "Error"]).assign(Comp="comp")
        dfi = pd.DataFrame(self.dat[1].T, columns=["RT", "Error"]).assign(Comp="incomp")
        df = pd.concat([dfc, dfi])

        self.caf = df.groupby(["Comp"]).apply(caffun, self.n_caf).reset_index()

    def _calc_delta_values(self):
        """Calculate compatibility effect + delta values for correct trials."""

        nbin = np.arange(1, self.n_delta + 1)

        # alphap, betap values to match R quantile 5 (see scipy.stats.mstats.mquantiles)
        mean_comp = mquantiles(
            self.dat[0][0],
            np.linspace(0, 1, self.n_delta + 2)[1:-1],
            alphap=0.5,
            betap=0.5,
        )
        mean_incomp = mquantiles(
            self.dat[1][0],
            np.linspace(0, 1, self.n_delta + 2)[1:-1],
            alphap=0.5,
            betap=0.5,
        )

        mean_bin = (mean_comp + mean_incomp) / 2
        mean_effect = mean_incomp - mean_comp

        dat = np.array([nbin, mean_comp, mean_incomp, mean_bin, mean_effect]).T

        self.delta = pd.DataFrame(
            dat, columns=["bin", "mean_comp", "mean_incomp", "mean_bin", "mean_effect"]
        )

    @staticmethod
    def rand_beta(lim=(0, 1), shape=3, n_trls=1):
        """Return random vector between limits weighted by beta function."""
        x = np.random.beta(shape, shape, n_trls)
        x = x * (lim[1] - lim[0]) + lim[0]

        return x

    def plot(
        self,
        fig_type="summary1",
        label_fontsize=12,
        tick_fontsize=10,
        hspace=0.5,
        wspace=0.5,
        **kwargs
    ):
        """Plot"""
        if fig_type == "summary1" and not self.full_data:
            fig_type = "summary2"

        if fig_type == "summary1":
            self._plot_summary1(label_fontsize, tick_fontsize, hspace, wspace, **kwargs)
        elif fig_type == "summary2":
            self._plot_summary2(label_fontsize, tick_fontsize, hspace, wspace, **kwargs)
        elif fig_type == "summary3":
            self._plot_summary3(label_fontsize, tick_fontsize, hspace, wspace, **kwargs)

    def _plot_summary1(self, label_fontsize, tick_fontsize, hspace, wspace, **kwargs):

        # upper left panel (activation)
        plt.subplot2grid((6, 4), (0, 0), rowspan=3, colspan=2)
        self.plot_activation(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # lower left panel (trials)
        plt.subplot2grid((6, 4), (3, 0), rowspan=3, colspan=2)
        self.plot_trials(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # upper right (left) panel (PDF)
        plt.subplot2grid((6, 4), (0, 2), rowspan=2)
        self.plot_pdf(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # upper right (right) panel (CDF)
        plt.subplot2grid((6, 4), (0, 3), rowspan=2)
        self.plot_cdf(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # middle right panel (CAF)
        plt.subplot2grid((6, 4), (2, 2), rowspan=2, colspan=2)
        self.plot_caf(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # bottom right panel (delta)
        plt.subplot2grid((6, 4), (4, 2), rowspan=2, colspan=2)
        self.plot_delta(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        plt.show(block=False)

    def _plot_summary2(self, label_fontsize, tick_fontsize, hspace, wspace, **kwargs):

        # upper right (left) panel (PDF)
        plt.subplot2grid((3, 2), (0, 0))
        self.plot_pdf(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # upper right (eight) panel (CDF)
        plt.subplot2grid((3, 2), (0, 1))
        self.plot_cdf(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # middle left panel
        plt.subplot2grid((3, 2), (1, 0), colspan=2)
        self.plot_caf(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # bottom right panel
        plt.subplot2grid((3, 2), (2, 0), colspan=2)
        self.plot_delta(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        plt.show(block=False)

    def _plot_summary3(self, label_fontsize, tick_fontsize, hspace, wspace, **kwargs):

        # upper right (left) panel (PDF)
        plt.subplot2grid((3, 1), (0, 0))
        self.plot_rt_correct(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # upper right (eight) panel (CDF)
        plt.subplot2grid((3, 1), (1, 0))
        self.plot_er(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # middle left panel
        plt.subplot2grid((3, 1), (2, 0))
        self.plot_rt_error(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        plt.show(block=False)

    def plot_activation(
        self,
        show=True,
        xlim=None,
        ylim=None,
        xlabel="Time (ms)",
        ylabel="E[X(t)]",
        label_fontsize=12,
        tick_fontsize=10,
        cond_labels=("Compatible", "Incompatible"),
        legend_position="best",
        colors=("green", "red"),
        **kwargs
    ):
        """Plot activation."""

        if not self.xt:
            print("Plotting activation function requires full_data=True")
            return

        plt.plot(self.eq4, "k-")
        plt.plot(self.eq4 * -1, "k--")
        plt.plot(self.xt[0], color=colors[0], label=cond_labels[0], **kwargs)
        plt.plot(self.xt[1], color=colors[1], label=cond_labels[1], **kwargs)
        plt.plot(np.cumsum(np.repeat(self.drc, self.t_max)), color="black", **kwargs)

        xlim = xlim or [0, self.t_max]
        ylim = ylim or [-self.bnds - 20, self.bnds + 20]
        self._plot_bounds()
        _adjust_plt(xlim, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if legend_position is not None:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def plot_trials(
        self,
        show=True,
        xlim=None,
        ylim=None,
        xlabel="Time (ms)",
        ylabel="X(t)",
        label_fontsize=12,
        tick_fontsize=10,
        cond_labels=("Compatible", "Incompatible"),
        legend_position="upper right",
        colors=("green", "red"),
        **kwargs
    ):
        """Plot individual trials."""

        if not self.xt:
            print("Plotting individual trials function requires full_data=True")
            return

        for trl in range(self.n_trls_data):
            if trl == 0:
                labels = cond_labels
            else:
                labels = [None, None]
            idx = np.where(np.abs(self.dat_trials[0][trl, :]) >= self.bnds)[0][0]
            plt.plot(
                self.dat_trials[0][trl][0:idx],
                color=colors[0],
                label=labels[0],
                **kwargs,
            )
            idx = np.where(np.abs(self.dat_trials[1][trl, :]) >= self.bnds)[0][0]
            plt.plot(
                self.dat_trials[1][trl][0:idx],
                color=colors[1],
                label=labels[1],
                **kwargs,
            )

        xlim = xlim or [0, self.t_max]
        ylim = ylim or [-self.bnds - 20, self.bnds + 20]
        self._plot_bounds()
        _adjust_plt(xlim, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def _plot_bounds(self):
        plt.axhline(y=self.bnds, color="black", linestyle="--")
        plt.axhline(y=-self.bnds, color="black", linestyle="--")

    def plot_pdf(
        self,
        show=True,
        xlim=None,
        ylim=None,
        xlabel="Time (ms)",
        ylabel="PDF",
        label_fontsize=12,
        tick_fontsize=10,
        cond_labels=("Compatible", "Incompatible"),
        legend_position="upper right",
        colors=("green", "red"),
        **kwargs
    ):
        """Plot PDF."""

        comp_pdf, axes1 = fastKDE.pdf(self.dat[0][0])
        incomp_pdf, axes2 = fastKDE.pdf(self.dat[1][0])

        plt.plot(axes1, comp_pdf, color=colors[0], label=cond_labels[0], **kwargs)
        plt.plot(axes2, incomp_pdf, color=colors[1], label=cond_labels[1], **kwargs)

        xlim = xlim or [0, self.t_max]
        ylim = ylim or [0, 0.01]
        _adjust_plt(xlim, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def plot_cdf(
        self,
        show=True,
        xlim=None,
        ylim=(0, 1.0),
        xlabel="Time (ms)",
        ylabel="CDF",
        label_fontsize=12,
        tick_fontsize=10,
        cond_labels=("Compatible", "Incompatible"),
        legend_position="lower right",
        colors=("green", "red"),
        **kwargs
    ):
        """Plot CDF."""

        comp_pdf, axes1 = fastKDE.pdf(self.dat[0][0])
        incomp_pdf, axes2 = fastKDE.pdf(self.dat[1][0])

        plt.plot(
            axes1,
            np.cumsum(comp_pdf) * np.diff(axes1)[0:1],
            color=colors[0],
            label=cond_labels[0],
            **kwargs,
        )
        plt.plot(
            axes2,
            np.cumsum(incomp_pdf) * np.diff(axes2)[0:1],
            color=colors[1],
            label=cond_labels[1],
            **kwargs,
        )

        xlim = xlim or [0, self.t_max]
        _adjust_plt(xlim, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if legend_position is not None:
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
        xlim=None,
        ylim=None,
        xlabel="Time (ms)",
        ylabel=r"$\Delta$",
        label_fontsize=12,
        tick_fontsize=10,
        **kwargs
    ):
        """Plot reaction-time delta plots."""

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        plt.plot(self.delta["mean_bin"], self.delta["mean_effect"], **kwargs)

        xlim = xlim or [
            np.min(self.delta.mean_bin) - 100,
            np.max(self.delta.mean_bin) + 100,
        ]
        ylim = ylim or [
            np.min(self.delta.mean_effect) - 25,
            np.max(self.delta.mean_effect) + 25,
        ]
        _adjust_plt(xlim, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if show:
            plt.show(block=False)

    def _dr(self):
        if self.var_dr:
            return self.rand_beta(self.dr_lim, self.dr_shape, self.n_trls)
        return np.ones(self.n_trls) * self.drc

    def _sp(self):
        if self.var_sp:
            return self.rand_beta(
                (self.sp_lim[0], self.sp_lim[1]), self.sp_shape, self.n_trls
            )
        return np.zeros(self.n_trls)

    def plot_rt_correct(
        self,
        show=True,
        ylim=None,
        xlabel=None,
        ylabel="RT Correct [ms]",
        label_fontsize=12,
        tick_fontsize=10,
        cond_labels=("Compatible", "Incompatible"),
        **kwargs
    ):
        """Plot correct RT's."""

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        plt.plot(cond_labels, self.summary["rt_cor"], **kwargs)

        if ylim is None:
            ylim = [
                np.min(self.summary["rt_cor"]) - 100,
                np.max(self.summary["rt_cor"]) + 100,
            ]

        plt.margins(x=0.5)
        _adjust_plt(None, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if show:
            plt.show(block=False)

    def plot_er(
        self,
        show=True,
        ylim=None,
        xlabel=None,
        ylabel="Error Rate [%]",
        label_fontsize=12,
        tick_fontsize=10,
        cond_labels=("Compatible", "Incompatible"),
        **kwargs
    ):
        """Plot error rate"""

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
        ylabel="RT Error [ms]",
        label_fontsize=12,
        tick_fontsize=10,
        cond_labels=("Compatible", "Incompatible"),
        **kwargs
    ):
        """Plot error RT's."""

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        plt.plot(cond_labels, self.summary["rt_err"], **kwargs)

        if ylim is None:
            ylim = [
                np.min(self.summary["rt_err"]) - 100,
                np.max(self.summary["rt_err"]) + 100,
            ]

        plt.margins(x=0.5)
        _adjust_plt(None, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if show:
            plt.show(block=False)


@jit(nopython=True, parallel=True)
def _run_simulation(
    drc, sp, dr, t_max, sigma, res_dist_type, res_mean, res_sd, bnds, n_trls
):

    dat = np.vstack((np.ones(n_trls) * t_max, np.zeros(n_trls)))
    if res_dist_type == 1:
        res_dist = np.random.normal(res_mean, res_sd, n_trls)
    else:
        width = max([0.01, np.sqrt((res_sd * res_sd / (1 / 12)))])
        res_dist = np.random.uniform(res_mean - width, res_mean + width, n_trls)

    for trl in prange(n_trls):
        trl_xt = sp[trl]
        for tp in range(0, t_max):
            trl_xt += drc[tp] + dr[trl] + (sigma * np.random.randn())
            if np.abs(trl_xt) >= bnds:
                dat[0, trl] = tp + max(0, res_dist[trl])
                dat[1, trl] = trl_xt < 0.0
                break

    return dat


@jit(nopython=True, parallel=True)
def _run_simulation_full(
    drc,
    sp,
    dr,
    t_max,
    sigma,
    res_dist_type,
    res_mean,
    res_sd,
    bnds,
    n_trls,
    n_trls_data,
):

    dat = np.vstack((np.ones(n_trls) * t_max, np.zeros(n_trls)))
    if res_dist_type == 1:
        res_dist = np.random.normal(res_mean, res_sd, n_trls)
    else:
        width = max([0.01, np.sqrt((res_sd * res_sd / (1 / 12)))])
        res_dist = np.random.uniform(res_mean - width, res_mean + width, n_trls)

    activation = np.zeros(t_max)
    trials = np.zeros((n_trls_data, t_max))
    for trl in prange(n_trls):

        rand_nums = np.random.randn(1, t_max)
        xt = drc + dr[trl] + (sigma * rand_nums)

        # variable starting point
        xt[0] += sp[trl]

        # cumulate activation over time
        xt = np.cumsum(xt)

        for tp in range(len(xt)):
            if np.abs(xt[tp]) >= bnds:
                dat[0, trl] = tp + max(0, res_dist[trl])
                dat[1, trl] = xt[tp] < 0.0
                break

        activation += xt
        if trl < n_trls_data:
            trials[trl, :] = xt

    activation /= n_trls

    return activation, trials, dat


def _adjust_plt(xlim, ylim, xlabel, ylabel, label_fontsize, tick_fontsize):
    """Internal function to adjust some plot properties."""
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.yticks(fontsize=tick_fontsize)
