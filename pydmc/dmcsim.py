"""
DMC model simulation detailed in  Ulrich, R., Schröter, H., Leuthold, H.,
& Birngruber, T. (2015). Automatic and controlled stimulus processing
in conflict tasks: Superimposed diffusion processes and delta functions.
Cognitive Psychology, 78, 148-174.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from fastkde import fastKDE
from numba import jit, prange
from scipy.stats.mstats import mquantiles


@dataclass
class DmcParameters:
    """
    DMC Parameters
    ----------
    amp: int/float, optional
        amplitude of automatic activation
    tau: int/float, optional
        time to peak automatic activation
    aa_shape: int/float, optional
        shape parameter of automatic activation
    drc: int/float, optional
        drift rate of controlled processes
    sigma: int/float, optional
        diffusion constant
    bnds: int/float, optional
        +- response barrier
    res_dist: int, optional
        non-decisional component distribution (1=normal, 2=uniform)
    res_mean: int/float, optional
        mean of non-decisional component
    res_sd: int/float, optional
        standard deviation of non-decisional component
    t_max: int, optional
        number of time points per trial
    var_sp: bool, optional
        variable starting point
    sp_lim: tuple, optional
        limiit range of distribution of starting point
    sp_shape: int/float, optional
        shape parameter of starting point distribution
    var_dr: bool, optional
        variable drift rate
    dr_lim: tuple, optional
        limit range of distribution of drift rate
    dr_shape: int, optional
        shape parameter of drift rate
    """
    amp: float = 20
    tau: float = 30
    aa_shape: float = 2
    drc: float = 0.5
    sigma: float = 4
    bnds: float = 75
    res_dist: int = 1
    res_mean: float = 300
    res_sd: float = 30
    t_max: int = 1000
    var_sp: bool = False
    sp_lim: tuple = (-75, 75)
    sp_shape: float = 3
    var_dr: bool = False
    dr_lim: tuple = (0.1, 0.7)
    dr_shape: float = 3


class DmcSim:
    """DMC Simulation."""
    def __init__(
        self,
        prms=DmcParameters(),
        n_trls=100000,
        n_caf=5,
        n_delta=19,
        p_delta=None,
        t_delta=1,
        full_data=False,
        n_trls_data=5,
        run_simulation=True,
        plt_figs=False,
    ):
        """
        n_trls: int (1000 to 100000 (default)), optional
            number of trials
       plt_figs: bool, optional
            plot figures
        n_caf: range, optional
            caf bins
        n_delta: range, optional
            delta reaction time bins
        p_delta: array, optional
            delta percentiles
        t_delta: int, optional
            type of delta calculation (1 = percentile, 2 = percentil bin average)
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
        >>> from pydmc.dmcsim import DmcSim, DmcParameters
        >>> dmc_sim = DmcSim(full_data=True)
        >>> dmc_sim.plot()                 # Fig 3
        >>> dmc_sim = DmcSim()
        >>> dmc_sim.plot()                 # Fig 3 (part)
        >>> dmc_sim = DmcSim(DmcParameters(tau = 150))
        >>> dmc_sim.plot()                 # Fig 4
        >>> dmc_sim = DmcSim(DmcParameters(tau = 90))
        >>> dmc_sim.plot()                 # Fig 5
        >>> dmc_sim = DmcSim(DmcParameters(var_sp = True))
        >>> dmc_sim.plot()                 # Fig 6
        >>> dmc_sim = DmcSim(DmcParameters(var_dr = True))
        >>> dmc_sim.plot()                 # Fig 7
        """

        self.prms = prms
        self.n_trls = n_trls
        self.n_caf = n_caf
        self.n_delta = n_delta
        self.p_delta = p_delta
        self.t_delta = t_delta
        self.full_data = full_data
        self.n_trls_data = n_trls_data

        self.tim = None
        self.eq4 = None
        self.dat = None
        self.dat_trials = None
        self.xt = None
        self.summary = None
        self.caf = None
        self.delta = None

        if run_simulation:
            self.run_simulation()

        if plt_figs:
            self.plot()

    def run_simulation(self):
        """Run simulation."""

        self.tim = np.arange(1, self.prms.t_max + 1, 1)
        self.eq4 = (
            self.prms.amp
            * np.exp(-self.tim / self.prms.tau)
            * (np.exp(1) * self.tim / (self.prms.aa_shape - 1) / self.prms.tau)
            ** (self.prms.aa_shape - 1)
        )
        if self.full_data:
            self._run_simulation_full()
        else:
            self._run_simulation()

        self._calc_caf_values()
        self._calc_delta_values()
        self._results_summary()

    def _run_simulation(self):
        """Run simulation using numba."""

        self.dat = []
        for comp in (1, -1):

            dr, sp = self._dr(),  self._sp()
            drc = comp * self.eq4 * ((self.prms.aa_shape - 1) / self.tim - 1 / self.prms.tau)

            self.dat.append(
                _run_simulation(
                    drc,
                    sp,
                    dr,
                    self.prms.t_max,
                    self.prms.sigma,
                    self.prms.res_dist,
                    self.prms.res_mean,
                    self.prms.res_sd,
                    self.prms.bnds,
                    self.n_trls,
                )
            )

    def _run_simulation_full(self):
        """Run simulation using numba."""

        self.xt = []
        self.dat_trials = []
        self.dat = []
        for comp in (1, -1):

            dr, sp = self._dr(), self._sp()
            drc = comp * self.eq4 * ((self.prms.aa_shape - 1) / self.tim - 1 / self.prms.tau)

            activation, trials, dat = _run_simulation_full(
                drc,
                sp,
                dr,
                self.prms.t_max,
                self.prms.sigma,
                self.prms.res_dist,
                self.prms.res_mean,
                self.prms.res_sd,
                self.prms.bnds,
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

        if self.t_delta == 1:

            if self.p_delta is not None:
                percentiles = self.p_delta
            else:
                percentiles = np.linspace(0, 1, self.n_delta + 2)[1:-1]

            # alphap, betap values to match R quantile 5 (see scipy.stats.mstats.mquantiles)
            mean_bins = np.array([mquantiles(
                self.dat[comp][0],
                percentiles,
                alphap=0.5, betap=0.5,
            ) for comp in (0, 1)])

        elif self.t_delta == 2:

            if self.p_delta is not None:
                percentiles = [0] + self.p_delta + [1]
            else:
                percentiles = np.linspace(0, 1, self.n_delta + 1)

            mean_bins = np.zeros((2, len(percentiles)-1))
            for comp in (0, 1):
                bin_values = mquantiles(
                    self.dat[comp][0],
                    percentiles,
                    alphap=0.5, betap=0.5,
                )

                tile = np.digitize(self.dat[comp][0], bin_values)
                mean_bins[comp, :] = np.array([self.dat[comp][0][tile == i].mean() for i in range(1, len(bin_values))])

        mean_bin = mean_bins.mean(axis=0)
        mean_effect = mean_bins[1, :] - mean_bins[0, :]

        dat = np.array([range(1, len(mean_bin)+1), mean_bins[0, :], mean_bins[1, :], mean_bin, mean_effect]).T

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
        """Plot

        Parameters
        ----------
        fig_type
        label_fontsize
        tick_fontsize
        hspace
        wspace
        kwargs
        """
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
        """Plot activation.

        Parameters
        ----------
        show
        xlim
        ylim
        xlabel
        ylabel
        label_fontsize
        tick_fontsize
        cond_labels
        legend_position
        colors
        kwargs
        """

        if not self.xt:
            print("Plotting activation function requires full_data=True")
            return

        plt.plot(self.eq4, "k-")
        plt.plot(self.eq4 * -1, "k--")
        plt.plot(self.xt[0], color=colors[0], label=cond_labels[0], **kwargs)
        plt.plot(self.xt[1], color=colors[1], label=cond_labels[1], **kwargs)
        plt.plot(np.cumsum(np.repeat(self.prms.drc, self.prms.t_max)), color="black", **kwargs)

        xlim = xlim or [0, self.prms.t_max]
        ylim = ylim or [-self.prms.bnds - 20, self.prms.bnds + 20]
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
        """Plot individual trials.

        Parameters
        ----------
        show
        xlim
        ylim
        xlabel
        ylabel
        label_fontsize
        tick_fontsize
        cond_labels
        legend_position
        colors
        kwargs
        """

        if not self.xt:
            print("Plotting individual trials function requires full_data=True")
            return

        for trl in range(self.n_trls_data):
            if trl == 0:
                labels = cond_labels
            else:
                labels = [None, None]
            idx = np.where(np.abs(self.dat_trials[0][trl, :]) >= self.prms.bnds)[0][0]
            plt.plot(
                self.dat_trials[0][trl][0:idx],
                color=colors[0],
                label=labels[0],
                **kwargs,
            )
            idx = np.where(np.abs(self.dat_trials[1][trl, :]) >= self.prms.bnds)[0][0]
            plt.plot(
                self.dat_trials[1][trl][0:idx],
                color=colors[1],
                label=labels[1],
                **kwargs,
            )

        xlim = xlim or [0, self.prms.t_max]
        ylim = ylim or [-self.prms.bnds - 20, self.prms.bnds + 20]
        self._plot_bounds()
        _adjust_plt(xlim, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def _plot_bounds(self):
        plt.axhline(y=self.prms.bnds, color="black", linestyle="--")
        plt.axhline(y=-self.prms.bnds, color="black", linestyle="--")

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
        """Plot PDF.

        Parameters
        ----------
        show
        xlim
        ylim
        xlabel
        ylabel
        label_fontsize
        tick_fontsize
        cond_labels
        legend_position
        colors
        kwargs
        """

        comp_pdf, axes1 = fastKDE.pdf(self.dat[0][0])
        incomp_pdf, axes2 = fastKDE.pdf(self.dat[1][0])

        plt.plot(axes1, comp_pdf, color=colors[0], label=cond_labels[0], **kwargs)
        plt.plot(axes2, incomp_pdf, color=colors[1], label=cond_labels[1], **kwargs)

        xlim = xlim or [0, self.prms.t_max]
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
        """Plot CDF.

        Parameters
        ----------
        show
        xlim
        ylim
        xlabel
        ylabel
        label_fontsize
        tick_fontsize
        cond_labels
        legend_position
        colors
        kwargs
        """

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

        xlim = xlim or [0, self.prms.t_max]
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
        """Plot CAF.

        Parameters
        ----------
        show
        ylim
        xlabel
        ylabel
        label_fontsize
        tick_fontsize
        cond_labels
        legend_position
        colors
        kwargs
        """

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
        """Plot reaction-time delta plots.

        Parameters
        ----------
        show
        xlim
        ylim
        xlabel
        ylabel
        label_fontsize
        tick_fontsize
        kwargs
        """

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
        if self.prms.var_dr:
            return self.rand_beta(self.prms.dr_lim, self.prms.dr_shape, self.n_trls)
        return np.ones(self.n_trls) * self.prms.drc

    def _sp(self):
        if self.prms.var_sp:
            return self.rand_beta(
                (self.prms.sp_lim[0], self.prms.sp_lim[1]), self.prms.sp_shape, self.n_trls
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
        """Plot correct RT's.

        Parameters
        ----------
        show
        ylim
        xlabel
        ylabel
        label_fontsize
        tick_fontsize
        cond_labels
        kwargs
        """

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
        """Plot error rate

        Parameters
        ----------
        show
        ylim
        xlabel
        ylabel
        label_fontsize
        tick_fontsize
        cond_labels
        kwargs
        """

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
        """Plot error RT's.

        Parameters
        ----------
        show
        ylim
        xlabel
        ylabel
        label_fontsize
        tick_fontsize
        cond_labels
        kwargs
        """

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
