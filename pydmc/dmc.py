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
        self.dat = [None, None]
        self.dat_trials = [None, None]
        self.xt = [None, None]

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
        for idx, comp in enumerate([1, -1]):
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

            self.dat[idx] = np.vstack(
                (
                    rt + np.random.normal(self.res_mean, self.res_sd, self.n_trls),
                    xt[np.arange(len(xt)), rt - 1] < self.bnds,
                )
            )
            self.dat_trials[idx] = xt[0 : self.n_trls_data]
            self.xt[idx] = xt.mean(0)

        self._calc_caf_values()
        self._calc_delta_values()
        self._results_summary()

    def _run_simulation_numba(self):
        """Run simulation using numba."""

        for idx, comp in enumerate([1, -1]):

            dr = self._dr()
            sp = self._sp()
            drc = comp * self.eq4 * ((self.aa_shape - 1) / self.tim - 1 / self.tau)

            self.dat[idx] = _run_simulation_numba(
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

        self._calc_caf_values()
        self._calc_delta_values()
        self._results_summary()

    def _results_summary(self):
        """Create results summary table."""

        res = [
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

        self.res = pd.DataFrame(
            res,
            index=["comp", "incomp"],
            columns=["rtCorr", "sdCorr", "perErr", "rtErr", "sdRtErr"],
        )

        print(self.res)

    def _calc_caf_values(self):
        """Calculate conditional accuracy functions."""

        bc = np.digitize(
            self.dat[0][0],
            np.percentile(self.dat[0][0], np.linspace(0, 100, self.n_caf + 1)[:-1]),
        )
        bi = np.digitize(
            self.dat[1][0],
            np.percentile(self.dat[1][0], np.linspace(0, 100, self.n_caf + 1)[:-1]),
        )
        self.caf = np.zeros((2, len(np.unique(bc))))
        for bidx, bedge in enumerate(np.unique(bc)):

            idx = np.where(bc == bedge)
            self.caf[0][bidx] = 1 - (sum(self.dat[0][1][idx]) / len(idx[0]))

            idx = np.where(bi == bedge)
            self.caf[1][bidx] = 1 - (sum(self.dat[1][1][idx]) / len(idx[0]))

    def _calc_delta_values(self):
        """Calculate compatibility effect + delta values for correct trials."""

        bc = np.percentile(self.dat[0][0], np.linspace(0, 100, self.n_delta + 2)[1:-1])
        bi = np.percentile(self.dat[1][0], np.linspace(0, 100, self.n_delta + 2)[1:-1])

        self.time_delta = (bc + bi) / 2
        self.effect_delta = bi - bc

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

        # middle riight panel (CAF)
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

        if self.xt[0] is None:
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
        plt.yticks((-self.bnds, self.bnds))
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

        if self.xt[0] is None:
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
        plt.yticks((-self.bnds, self.bnds))
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
        plt.plot(self.caf[0], cols[0], linestyle="-", marker="o")
        plt.plot(self.caf[1], cols[1], linestyle="-", marker="o")

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

        plt.plot(self.time_delta, self.effect_delta, "ko-", markersize=4)

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


if __name__ == "__main__":
    dmc = DmcSim()
