"""
DMC model simulation detailed in  Ulrich, R., Schröter, H., Leuthold, H.,
& Birngruber, T. (2015). Automatic and controlled stimulus processing
in conflict tasks: Superimposed diffusion processes and delta functions.
Cognitive Psychology, 78, 148-174.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from numba import jit, prange
from scipy.stats.mstats import mquantiles
from pydmc.dmcplot import DmcPlot


@dataclass
class DmcParameters:
    """
    DMC Parameters
    ----------
    amp: int/float, optional
        amplitude of automatic activation
    tau: int/float, optional
        time to peak automatic activation
    drc: int/float, optional
        drift rate of controlled processes
    bnds: int/float, optional
        +- response barrier
    res_dist: int, optional
        non-decisional component distribution (1=normal, 2=uniform)
    res_mean: int/float, optional
        mean of non-decisional component
    res_sd: int/float, optional
        standard deviation of non-decisional component
    aa_shape: int/float, optional
        shape parameter of automatic activation
    sp_shape: int/float, optional
        shape parameter of starting point distribution
    sp_biad: int/float, optional
        starting point bias
    sigma: int/float, optional
        diffusion constant
    t_max: int, optional
        number of time points per trial
    sp_dist: int, optional
        starting point distribution (0 = constant, 1 = beta, 2 = uniform)
    sp_lim: tuple, optional
        limiit range of distribution of starting point
    dr_dist: int, optional
        drift rate distribution (0 = constant, 1 = beta, 2 = uniform)
    dr_lim: tuple, optional
        limit range of distribution of drift rate
    dr_shape: int, optional
        shape parameter of drift rate
    """

    amp: float = 20
    tau: float = 30
    drc: float = 0.5
    bnds: float = 75
    res_dist: int = 1
    res_mean: float = 300
    res_sd: float = 30
    aa_shape: float = 2
    sp_shape: float = 3
    sigma: float = 4
    t_max: int = 1000
    sp_dist: int = 0
    sp_lim: tuple = (-75, 75)
    sp_bias: float = 0.0
    dr_dist: int = 0
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
        n_trls: int 100000 (default), optional
            number of trials
        n_caf: range, optional
            caf bins
        n_delta: range, optional
            delta reaction time bins
        p_delta: array, optional
            delta percentiles
        t_delta: int, optional
            type of delta calculation (1 = percentile, 2 = percentile bin average)
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
        >>> dmc_sim.plot()      # Fig 3
        >>> dmc_sim = DmcSim()
        >>> dmc_sim.plot()      # Fig 3 (part)
        >>> dmc_sim = DmcSim(DmcParameters(tau = 150))
        >>> dmc_sim.plot()      # Fig 4
        >>> dmc_sim = DmcSim(DmcParameters(tau = 90))
        >>> dmc_sim.plot()      # Fig 5
        >>> dmc_sim = DmcSim(DmcParameters(sp_dist = 1))
        >>> dmc_sim.plot()      # Fig 6
        >>> dmc_sim = DmcSim(DmcParameters(dr_dist = 1))
        >>> dmc_sim.plot()      # Fig 7
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
        """Run DMC simulation."""

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

        self.dat = []
        for comp in (1, -1):

            drc = (
                comp
                * self.eq4
                * ((self.prms.aa_shape - 1) / self.tim - 1 / self.prms.tau)
            )
            dr, sp = self._dr(), self._sp()

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

        self.xt = []
        self.dat_trials = []
        self.dat = []
        for comp in (1, -1):

            drc = (
                comp
                * self.eq4
                * ((self.prms.aa_shape - 1) / self.tim - 1 / self.prms.tau)
            )
            dr, sp = self._dr(), self._sp()

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

        summary = []
        for comp in (0, 1):
            summary.append(
                [
                    np.round(np.mean(self.dat[comp][0][self.dat[comp][1] == 0])),
                    np.round(np.std(self.dat[comp][0][self.dat[comp][1] == 0])),
                    np.round(np.sum(self.dat[comp][1] / self.n_trls) * 100, 1),
                    np.round(np.mean(self.dat[comp][0][self.dat[comp][1] == 1])),
                    np.round(np.std(self.dat[comp][0][self.dat[comp][1] == 1])),
                ]
            )

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
            mean_bins = np.array(
                [
                    mquantiles(
                        self.dat[comp][0][self.dat[comp][1] == 0],
                        percentiles,
                        alphap=0.5,
                        betap=0.5,
                    )
                    for comp in (0, 1)
                ]
            )

        elif self.t_delta == 2:

            if self.p_delta is not None:
                percentiles = [0] + self.p_delta + [1]
            else:
                percentiles = np.linspace(0, 1, self.n_delta + 1)

            mean_bins = np.zeros((2, len(percentiles) - 1))
            for comp in (0, 1):
                bin_values = mquantiles(
                    self.dat[comp][0],
                    percentiles,
                    alphap=0.5,
                    betap=0.5,
                )

                tile = np.digitize(self.dat[comp][0], bin_values)
                mean_bins[comp, :] = np.array(
                    [
                        self.dat[comp][0][tile == i].mean()
                        for i in range(1, len(bin_values))
                    ]
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

        self.delta = pd.DataFrame(
            dat, columns=["bin", "mean_comp", "mean_incomp", "mean_bin", "mean_effect"]
        )

    @staticmethod
    def rand_beta(lim=(0, 1), shape=3.0, n_trls=1):
        """Return random vector between limits weighted by beta function."""
        return np.random.beta(shape, shape, n_trls) * (lim[1] - lim[0]) + lim[0]

    def _dr(self):
        if self.prms.dr_dist == 0:
            # constant between trial drift rate
            return np.ones(self.n_trls) * self.prms.drc
        if self.prms.dr_dist == 1:
            # between trial variablity in drift rate from beta distribution
            return self.rand_beta(self.prms.dr_lim, self.prms.dr_shape, self.n_trls)
        if self.prms.dr_dist == 2:
            # between trial variablity in drift rate from uniform
            return np.random.uniform(
                self.prms.dr_lim[0], self.prms.dr_lim[1], self.n_trls
            )

    def _sp(self):
        if self.prms.sp_dist == 0:
            # constant between trial starting point
            return np.zeros(self.n_trls) + self.prms.sp_bias
        if self.prms.sp_dist == 1:
            # between trial variablity in starting point from beta distribution
            return (
                self.rand_beta(
                    (self.prms.sp_lim[0], self.prms.sp_lim[1]),
                    self.prms.sp_shape,
                    self.n_trls,
                )
                + self.prms.sp_bias
            )
        if self.prms.sp_dist == 2:
            # between trial variablity in starting point from uniform distribution
            return (
                np.random.uniform(self.prms.sp_lim[0], self.prms.sp_lim[1], self.n_trls)
                + self.prms.sp_bias
            )

    def plot(self, **kwargs):
        """Plot."""
        DmcPlot(self).plot(**kwargs)

    def plot_activation(self, **kwargs):
        """Plot activation."""
        DmcPlot(self).plot_activation(**kwargs)

    def plot_trials(self, **kwargs):
        """Plot trials."""
        DmcPlot(self).plot_trials(**kwargs)

    def plot_pdf(self, **kwargs):
        """Plot pdf."""
        DmcPlot(self).plot_pdf(**kwargs)

    def plot_cdf(self, **kwargs):
        """Plot cdf."""
        DmcPlot(self).plot_cdf(**kwargs)

    def plot_caf(self, **kwargs):
        """Plot caf."""
        DmcPlot(self).plot_caf(**kwargs)

    def plot_delta(self, **kwargs):
        """Plot delta."""
        DmcPlot(self).plot_delta(**kwargs)

    def plot_rt_correct(self, **kwargs):
        """Plot rt correct."""
        DmcPlot(self).plot_rt_correct(**kwargs)

    def plot_er(self, **kwargs):
        """Plot er."""
        DmcPlot(self).plot_er(**kwargs)

    def plot_rt_error(self, **kwargs):
        """Plot rt error."""
        DmcPlot(self).plot_rt_error(**kwargs)


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
