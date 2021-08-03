"""
DMC model simulation detailed in  Ulrich, R., Schröter, H., Leuthold, H.,
& Birngruber, T. (2015). Automatic and controlled stimulus processing
in conflict tasks: Superimposed diffusion processes and delta functions.
Cognitive Psychology, 78, 148-174.
"""
import copy
import glob
import inspect
import pkg_resources
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from fastkde import fastKDE
from itertools import product
from numba import jit, prange
from scipy.stats.mstats import mquantiles
from scipy.optimize import minimize, differential_evolution
from typing import Union


@dataclass
class Prms:
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
    res_mean: int/float, optional
        mean of non-decisional component
    res_sd: int/float, optional
        standard deviation of non-decisional component
    aa_shape: int/float, optional
        shape parameter of automatic activation
    sp_shape: int/float, optional
        shape parameter of starting point distribution
    sp_bias: int/float, optional
        starting point bias
    sigma: int/float, optional
        diffusion constant
    res_dist: int, optional
        non-decisional component distribution (1=normal, 2=uniform)
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
    res_mean: float = 300
    res_sd: float = 30
    aa_shape: float = 2
    sp_shape: float = 3
    sigma: float = 4
    res_dist: int = 1
    t_max: int = 1000
    sp_dist: int = 0
    sp_lim: tuple = (-75, 75)
    sp_bias: float = 0.0
    dr_dist: int = 0
    dr_lim: tuple = (0.1, 0.7)
    dr_shape: float = 3


class Sim:
    """DMC Simulation."""

    def __init__(
        self,
        prms: Prms = Prms(),
        n_trls: int = 100_000,
        n_caf: int = 5,
        n_delta: int = 19,
        p_delta: tuple = (),
        t_delta: int = 1,
        full_data: bool = False,
        n_trls_data: int = 5,
        run_simulation: bool = True,
    ):
        """
        n_trls: int (100000), optional
            number of trials
        n_caf: int (5), optional
            number of caf bins
        n_delta: int (19), optional
            number of delta reaction time bins
        p_delta: tuple (), optional
            alternative to n_delta by directly specifying required percentile values (values between 0-1)
        t_delta: int 1, optional
            type of delta calculation (1=direct percentile points, 2=percentile bounds (tile) average)
        full_data: bool True, optional
            run simulation to t_max to calculate activation NB. only required when plotting activation function
            or inddividual trials
        n_trls_data: int 5, optional
            number of individual trials to store
        run_simulation: bool True, optional
            run simulation

        Returns
        -------
        - DmcSim object

        Notes
        -----
        - Tested with Python 3.9

        Examples
        --------
        >>> import pydmc
        >>> dmc_sim = pydmc.Sim(full_data=True)
        >>> dmc_sim.plot.summary()      # Fig 3
        >>> dmc_sim = pydmc.Sim()
        >>> dmc_sim.plot.summary()      # Fig 3
        >>> dmc_sim = pydmc.Sim(pydmc.Prms(tau = 150))
        >>> dmc_sim.plot.summary()      # Fig 4
        >>> dmc_sim = pydmc.Sim(pydmc.Prms(tau = 90))
        >>> dmc_sim.plot.summary()      # Fig 5
        >>> dmc_sim = pydmc.Sim(pydmc.Prms(sp_dist = 1))
        >>> dmc_sim.plot.summary()      # Fig 6
        >>> dmc_sim = pydmc.Sim(pydmc.Prms(dr_dist = 1))
        >>> dmc_sim.plot.summary()      # Fig 7
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
        self.data = None
        self.data_trials = None
        self.xt = None
        self.summary = None
        self.caf = None
        self.delta = None
        self.plot: Plot = Plot(self)

        if run_simulation:
            self.run_simulation()

    def run_simulation(self) -> None:
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

    def _run_simulation(self) -> None:

        self.data = []
        for comp in (1, -1):

            drc = (
                comp
                * self.eq4
                * ((self.prms.aa_shape - 1) / self.tim - 1 / self.prms.tau)
            )
            dr, sp = self._dr(), self._sp()

            self.data.append(
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

    def _run_simulation_full(self) -> None:

        self.xt = []
        self.data_trials = []
        self.data = []
        for comp in (1, -1):

            drc = (
                comp
                * self.eq4
                * ((self.prms.aa_shape - 1) / self.tim - 1 / self.prms.tau)
            )
            dr, sp = self._dr(), self._sp()

            activation, trials, data = _run_simulation_full(
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
            self.data_trials.append(trials)
            self.data.append(data)

    def _results_summary(self) -> None:
        """Create results summary table."""

        summary = []
        for comp in (0, 1):
            summary.append(
                [
                    np.round(np.mean(self.data[comp][0][self.data[comp][1] == 0])),
                    np.round(np.std(self.data[comp][0][self.data[comp][1] == 0])),
                    np.round(np.sum(self.data[comp][1] / self.n_trls) * 100, 1),
                    np.round(np.mean(self.data[comp][0][self.data[comp][1] == 1])),
                    np.round(np.std(self.data[comp][0][self.data[comp][1] == 1])),
                ]
            )

        self.summary = pd.DataFrame(
            summary,
            index=["comp", "incomp"],
            columns=["rt_cor", "sd_cor", "per_err", "rt_err", "sd_rt_err"],
        )

    def _calc_caf_values(self) -> None:
        """Calculate conditional accuracy functions."""

        def caffun(x, n):
            cafbin = np.digitize(
                x.loc[:, "RT"],
                np.percentile(x.loc[:, "RT"], np.linspace(0, 100, n + 1)),
            )
            x = x.assign(bin=cafbin)
            return pd.DataFrame((1 - x.groupby(["bin"])["Error"].mean())[:-1])

        # create temp pandas dataframe
        dfc = pd.DataFrame(self.data[0].T, columns=["RT", "Error"]).assign(Comp="comp")
        dfi = pd.DataFrame(self.data[1].T, columns=["RT", "Error"]).assign(
            Comp="incomp"
        )

        self.caf = (
            pd.concat([dfc, dfi])
            .groupby(["Comp"])
            .apply(caffun, self.n_caf)
            .reset_index()
            .pivot(index="bin", columns="Comp", values="Error")
            .reset_index()
            .rename_axis(None, axis=1)
            .assign(effect=lambda x: (x["comp"] - x["incomp"]) * 100)
        )

    # noinspection PyUnboundLocalVariable
    def _calc_delta_values(self) -> None:
        """Calculate compatibility effect + delta values for correct trials."""

        if self.t_delta == 1:

            if len(self.p_delta) != 0:
                percentiles = self.p_delta
            else:
                percentiles = np.linspace(0, 1, self.n_delta + 2)[1:-1]

            # alphap, betap values to match R quantile 5 (see scipy.stats.mstats.mquantiles)
            mean_bins = np.array(
                [
                    mquantiles(
                        self.data[comp][0][self.data[comp][1] == 0],
                        percentiles,
                        alphap=0.5,
                        betap=0.5,
                    )
                    for comp in (0, 1)
                ]
            )

        elif self.t_delta == 2:

            if len(self.p_delta) != 0:
                percentiles = (0,) + self.p_delta + (1,)
            else:
                percentiles = np.linspace(0, 1, self.n_delta + 1)

            mean_bins = np.zeros((2, len(percentiles) - 1))
            for comp in (0, 1):
                bin_values = mquantiles(
                    self.data[comp][0],
                    percentiles,
                    alphap=0.5,
                    betap=0.5,
                )

                tile = np.digitize(self.data[comp][0], bin_values)
                mean_bins[comp, :] = np.array(
                    [
                        self.data[comp][0][tile == i].mean()
                        for i in range(1, len(bin_values))
                    ]
                )

        mean_bin = mean_bins.mean(axis=0)
        mean_effect = mean_bins[1, :] - mean_bins[0, :]

        data = np.array(
            [
                range(1, len(mean_bin) + 1),
                mean_bins[0, :],
                mean_bins[1, :],
                mean_bin,
                mean_effect,
            ]
        ).T

        self.delta = pd.DataFrame(
            data, columns=["bin", "mean_comp", "mean_incomp", "mean_bin", "mean_effect"]
        )

    @staticmethod
    def rand_beta(
        lim: tuple = (0, 1), shape: float = 3.0, n_trls: int = 1
    ) -> np.ndarray:
        """Return random vector between limits weighted by beta function."""
        return np.random.beta(shape, shape, n_trls) * (lim[1] - lim[0]) + lim[0]

    def _dr(self) -> np.ndarray:
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

    def _sp(self) -> np.ndarray:
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


@jit(nopython=True, parallel=True)
def _run_simulation(
    drc, sp, dr, t_max, sigma, res_dist_type, res_mean, res_sd, bnds, n_trls
):

    data = np.vstack((np.ones(n_trls) * t_max, np.zeros(n_trls)))
    if res_dist_type == 1:
        res_dist = np.random.normal(res_mean, res_sd, n_trls)
    else:
        width = max([0.01, np.sqrt((res_sd * res_sd / (1 / 12)))])
        res_dist = np.random.uniform(res_mean - width, res_mean + width, n_trls)

    for trl in prange(n_trls):
        trl_xt = sp[trl]
        for tp in range(0, t_max):
            trl_xt += drc[tp] + dr[trl] + (sigma * np.random.randn())
            if np.abs(trl_xt) > bnds:
                data[0, trl] = tp + max(0, res_dist[trl])
                data[1, trl] = trl_xt < 0.0
                break

    return data


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

    data = np.vstack((np.ones(n_trls) * t_max, np.zeros(n_trls)))
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
            if np.abs(xt[tp]) > bnds:
                data[0, trl] = tp + max(0, res_dist[trl])
                data[1, trl] = xt[tp] < 0.0
                break

        activation += xt
        if trl < n_trls_data:
            trials[trl, :] = xt

    activation /= n_trls

    return activation, trials, data


def flanker_data() -> pd.DataFrame:
    """Load raw Flanker data from Ulrich et al. (2015)."""
    datafile = pkg_resources.resource_stream(__name__, "data/flankerData.csv")
    return pd.read_csv(datafile, sep="\t")


def simon_data() -> pd.DataFrame:
    """Load raw Simon task data from Ulrich et al. (2015)."""
    datafile = pkg_resources.resource_stream(__name__, "data/simonData.csv")
    return pd.read_csv(datafile, sep="\t")


class Ob:
    def __init__(
        self,
        data: Union[str, pd.DataFrame],
        n_caf: int = 5,
        n_delta: int = 19,
        p_delta: tuple = (),
        t_delta: int = 1,
        outlier: tuple = (200, 1200),
        columns: tuple = ("Subject", "Comp", "RT", "Error"),
        comp_coding: tuple = ("comp", "incomp"),
        error_coding: tuple = (0, 1),
        sep: str = "\t",
        skiprows: int = 0,
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
        self.plot: Plot = Plot(self)

    @staticmethod
    def read_data_files(data: str, sep: str = "\t", skiprows: int = 0) -> pd.DataFrame:
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

    def _columns(self) -> None:
        try:
            self.data = self.data[list(self.columns)]
        except KeyError:
            raise Exception("requested columns not in data!")
        if len(self.data.columns) != 4:
            raise Exception("data does not contain required/requested coluumns!")
        if not any(self.data.columns.values == ("Subject", "Comp", "RT", "Error")):
            self.data.columns = ("Subject", "Comp", "RT", "Error")

    def _comp_coding(self) -> None:
        if self.comp_coding != ("comp", "incomp"):
            self.data["Comp"] = np.where(
                self.data["Comp"] == self.comp_coding[0], "comp", "incomp"
            )

    def _error_coding(self) -> None:
        if self.error_coding != (0, 1):
            self.data["Error"] = np.where(
                self.data["Error"] == self.error_coding[0], 0, 1
            )

    def _outlier(self) -> None:
        self.data["outlier"] = np.where(
            (self.data["RT"] <= self.outlier[0]) | (self.data["RT"] >= self.outlier[1]),
            1,
            0,
        )

    def _aggregate_trials(self) -> None:
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

    def _aggregate_subjects(self) -> None:
        def aggfun(x):
            n_subjects = len(x["Subject"])
            if n_subjects == 1:
                new_cols = [
                    [
                        n_subjects,
                        np.nanmean(x["rt_cor"]),
                        np.nan,
                        np.nan,
                        np.nanmean(x["rt_err"]),
                        np.nan,
                        np.nan,
                        np.nanmean(x["per_err"]),
                        np.nan,
                        np.nan,
                    ]
                ]
            else:
                new_cols = [
                    [
                        n_subjects,
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

    def _calc_caf_values(self) -> None:
        """Calculate conditional accuracy functions."""

        def caffun(x, n: int) -> pd.DataFrame:
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
            .pivot(index=("Subject", "bin"), columns="Comp", values="Error")
            .reset_index()
            .rename_axis(None, axis=1)
            .assign(effect=lambda x: (x["comp"] - x["incomp"]) * 100)
        )

        self.caf = (
            self.caf_subject.groupby("bin").mean().reset_index().drop("Subject", axis=1)
        )

    def _calc_delta_values(self) -> None:
        """Calculate compatibility effect + delta values for correct trials."""

        # noinspection PyUnboundLocalVariable
        def deltafun(x: pd.DataFrame) -> pd.DataFrame:
            # filter trials
            x = x[(x.outlier == 0) & (x.Error == 0)].reset_index()

            if self.t_delta == 1:

                if len(self.p_delta) != 0:
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

                if len(self.p_delta) != 0:
                    percentiles = (0,) + self.p_delta + (1,)
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
            self.data.groupby(["Subject"]).apply(deltafun).reset_index()
        ).drop("level_1", axis=1)

        def aggfun(x: pd.DataFrame) -> pd.DataFrame:
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

    def select_subject(self, subject: int) -> pd.DataFrame:
        """Select subject"""
        return self.data[self.data.Subject == subject]


@dataclass
class PrmsFit:
    # start, min, max, fitted, initial grid search
    amp: tuple = (20, 0, 40, True, False)
    tau: tuple = (30, 5, 300, True, True)
    drc: tuple = (0.5, 0.1, 1.0, True, False)
    bnds: tuple = (75, 20, 150, True, False)
    res_mean: tuple = (300, 200, 800, True, False)
    res_sd: tuple = (30, 5, 100, True, False)
    aa_shape: tuple = (2, 1, 3, True, False)
    sp_shape: tuple = (3, 2, 4, True, False)
    sigma: tuple = (4, 1, 10, False, False)

    def set_start_values(self, **kwargs) -> None:
        self._set_values(0, **kwargs)

    def set_min_values(self, **kwargs) -> None:
        self._set_values(1, **kwargs)

    def set_max_values(self, **kwargs) -> None:
        self._set_values(2, **kwargs)

    def set_fitted_values(self, **kwargs) -> None:
        self._set_values(3, **kwargs)

    def set_grid_search_values(self, **kwargs) -> None:
        self._set_values(4, **kwargs)

    def _set_values(self, idx: int = 0, **kwargs) -> None:
        kwargs = {k: v for k, v in kwargs.items() if k in asdict(self).keys()}
        [
            setattr(
                self,
                k,
                tuple(getattr(self, k)[i] if i != idx else v for i in range(5)),
            )
            for k, v in kwargs.items()
        ]

    def dmc_prms(self, sp_dist: int = 1) -> Prms:
        return Prms(
            **self.dict(0), sp_dist=sp_dist, sp_lim=(-self.bnds[0], self.bnds[1])
        )

    def array(self, idx: int = 0) -> list:
        return [x[idx] for x in asdict(self).values() if x[-2]]

    def dict(self, idx: int = 0) -> dict:
        return {k: v[idx] for k, v in asdict(self).items() if v}

    def bounds(self) -> list:
        return [x[1:3] for x in asdict(self).values() if x[-2]]


class Fit:
    def __init__(
        self,
        res_ob: Ob,
        n_trls: int = 100_000,
        start_vals: PrmsFit = PrmsFit(),
        search_grid: bool = True,
        n_grid: int = 10,
        n_delta: int = 19,
        p_delta: tuple = (),
        t_delta: int = 1,
        n_caf: int = 5,
        cost_function="RMSE",
    ):
        self.res_ob = res_ob
        self.res_th = None
        self.fit = None
        self.n_trls = n_trls
        self.start_vals = start_vals
        self.search_grid = search_grid
        self.n_grid = n_grid
        self.dmc_prms = start_vals.dmc_prms()
        self.n_delta = n_delta
        self.p_delta = p_delta
        self.t_delta = t_delta
        self.n_caf = n_caf
        self.cost_function = self._assign_cost_function(cost_function)
        self.cost_value = np.Inf
        self.plot: None

    def _assign_cost_function(self, cost_function: str):
        if cost_function == "RMSE":
            return self.calculate_cost_value_rmse
        elif cost_function == "SPE":
            return self.calculate_cost_value_spe
        else:
            raise Exception("cost function not implemented!")

    def _search_grid(self) -> None:

        grid_space = {}
        for p in asdict(self.start_vals).items():
            if p[1][-1]:
                grid_space[p[0]] = np.linspace(p[1][1], p[1][2], self.n_grid)

        min_cost = np.Inf
        best_prms = self.dmc_prms
        combs = list(
            product(*grid_space.values())
        )  # TO DO: remove list but get length?
        for idx, comb in enumerate(combs):
            print(f"Searching grid combination {idx}/{len(combs)}")
            self.res_th.prms = Prms(**dict(zip(grid_space.keys(), comb)), sp_dist=1)
            self.res_th.run_simulation()
            cost = self.cost_function(self.res_th, self.res_ob)
            if cost < min_cost:
                min_cost, best_prms = cost, self.res_th.prms
        self.start_vals.set_start_values(**asdict(best_prms))
        self.dmc_prms = self.start_vals.dmc_prms()

    def fit_data(self, method: str = "nelder-mead", **kwargs) -> None:

        self.res_th = Sim(copy.deepcopy(self.dmc_prms))

        if method == "nelder-mead":
            if self.search_grid:
                self._search_grid()
            self._fit_data_neldermead(**kwargs)
        elif method == "differential_evolution":
            self._fit_data_differential_evolution(**kwargs)
        self.plot = PlotFit(self)

    def _fit_data_neldermead(self, **kwargs) -> None:
        kwargs.setdefault("maxiter", 500)
        self.fit = minimize(
            self._function_to_minimise,
            np.array(self.start_vals.array(0)),
            method="nelder-mead",
            bounds=self.start_vals.bounds(),
            options=kwargs,
        )

    def _fit_data_differential_evolution(self, **kwargs) -> None:
        kwargs.setdefault("maxiter", 100)
        kwargs.setdefault("polish", False)
        self.fit = differential_evolution(
            self._function_to_minimise,
            self.start_vals.bounds(),
            **kwargs,
        )

    def print_summary(self) -> None:
        """Print summary of DmcFit."""
        print(
            f"amp:{self.res_th.prms.amp:4.1f}",
            f"tau:{self.res_th.prms.tau:4.1f}",
            f"drc:{self.res_th.prms.drc:4.2f}",
            f"bnds:{self.res_th.prms.bnds:4.1f}",
            f"res_mean:{self.res_th.prms.res_mean:4.0f}",
            f"res_sd:{self.res_th.prms.res_sd:4.1f}",
            f"aa_shape:{self.res_th.prms.aa_shape:4.1f}",
            f"sp_shape:{self.res_th.prms.sp_shape:4.1f}",
            f"| cost={self.cost_value:.2f}",
        )

    def table_summary(self) -> pd.DataFrame:
        """Table summary."""
        return pd.DataFrame(
            [
                [
                    self.res_th.prms.amp,
                    self.res_th.prms.tau,
                    self.res_th.prms.drc,
                    self.res_th.prms.bnds,
                    self.res_th.prms.res_mean,
                    self.res_th.prms.res_sd,
                    self.res_th.prms.aa_shape,
                    self.res_th.prms.sp_shape,
                    self.res_th.prms.sigma,
                    self.cost_value,
                ]
            ],
            columns=[
                "amp",
                "tau",
                "drc",
                "bnds",
                "res_mean",
                "res_sd",
                "aa_shape",
                "sp_shape",
                "sigma",
                "cost",
            ],
            index=None,
        )

    def _function_to_minimise(self, x: list) -> float:

        self._update_parameters(x)
        self.res_th.run_simulation()
        self.cost_value = self.cost_function(self.res_th, self.res_ob)
        self.print_summary()

        return self.cost_value

    def _update_parameters(self, x: list) -> None:
        idx = 0
        for k in asdict(self.start_vals).keys():
            if getattr(self.start_vals, k)[-2]:
                setattr(self.res_th.prms, k, x[idx])
                idx += 1
        self.res_th.prms.sp_lim = (-self.res_th.prms.bnds, self.res_th.prms.bnds)

    @staticmethod
    def calculate_cost_value_rmse(res_th: Sim, res_ob: Ob) -> float:
        """calculate_cost_value_rmse

        Calculate Root Mean Square Error between simulated
        and observed data points

        Parameters
        ----------
        res_th
        res_ob
        """
        n_rt = len(res_th.delta) * 2
        n_err = len(res_th.caf) * 2

        cost_caf = np.sqrt(
            (1 / n_err)
            * np.sum(
                np.sum(
                    (res_th.caf[["comp", "incomp"]] - res_ob.caf[["comp", "incomp"]])
                    ** 2
                )
            )
        )
        cost_rt = np.sqrt(
            (1 / n_rt)
            * np.sum(
                np.sum(
                    (
                        res_th.delta[["mean_comp", "mean_incomp"]]
                        - res_ob.delta[["mean_comp", "mean_incomp"]]
                    )
                    ** 2
                )
            )
        )

        weight_rt = n_rt / (n_rt + n_err)
        weight_caf = (1 - weight_rt) * 1500

        return (weight_caf * cost_caf) + (weight_rt * cost_rt)

    @staticmethod
    def calculate_cost_value_spe(res_th: Sim, res_ob: Ob) -> float:
        """calculate_cost_calue_spe

        Calculate Squared Percentage Error between simulated
        and observed data points

        Parameters
        ---------
        res_th
        res_ob
        """
        cost_caf = np.sum(
            (
                (res_ob.caf.iloc[:, 1:3] - res_th.caf.iloc[:, 1:3])
                / res_ob.caf.iloc[:, 1:3]
            )
            ** 2
        ).sum()

        cost_rt = np.sum(
            (
                (res_ob.delta.iloc[:, 1:3] - res_th.delta.iloc[:, 1:3])
                / res_ob.delta.iloc[:, 1:3]
            )
            ** 2
        ).sum()

        return cost_rt + cost_caf


class FitSubjects:
    def __init__(self, res_ob: Ob):
        self.res_ob = res_ob
        self.subjects = np.unique(res_ob.summary_subject.Subject)
        self.fits = self._split_subjects()

    def _split_subjects(self) -> list[Fit]:
        return [
            copy.deepcopy(Fit(self.res_ob.select_subject(s))) for s in self.subjects
        ]

    def fit_data_neldermead(self, **kwargs) -> None:
        """Fit data using neldermead."""
        [f.fit_data_neldermead(**kwargs) for f in self.fits]

    def fit_data_differential_evolution(self, **kwargs) -> None:
        """Fit data using differential evolution."""
        [f.fit_data_differential_evolution(**kwargs) for f in self.fits]

    def print_summary(self) -> None:
        """Print summary of individual fits."""
        for idx, f in enumerate(self.fits):
            print(f"Subject: {idx+1}\t")
            f.print_summary()

    def table_summary(self) -> pd.DataFrame:
        """Combine tables of individual fits."""
        tables = []
        for idx, f in enumerate(self.fits):
            tmp_table = f.table_summary()
            tmp_table.insert(0, "Subject", idx + 1)
            tables.append(tmp_table)
        return pd.concat(tables)


class Plot:
    def __init__(self, res: Union[Sim, Ob]):

        assert isinstance(res, Sim) or isinstance(
            res, Ob
        ), "res must be of type 'Sim' or 'Ob'"

        self.res = res

    def summary(self, **kwargs):
        """Plot summary."""
        kwargs.setdefault("fig_type", "summary1")
        kwargs.setdefault("hspace", 0.5)
        kwargs.setdefault("wspace", 0.5)

        plt.figure(len(plt.get_fignums()) + 1)

        if isinstance(self.res, Sim):
            if kwargs["fig_type"] == "summary1" and not self.res.full_data:
                kwargs["fig_type"] = "summary2"
            if kwargs["fig_type"] == "summary1":
                self._summary1_res_th(**kwargs)
            elif kwargs["fig_type"] == "summary2":
                self._summary2_res_th(**kwargs)
            elif kwargs["fig_type"] == "summary3":
                self._summary3_res_th(**kwargs)
        else:
            self._summary1_res_ob(**kwargs)

    def _summary1_res_th(self, **kwargs):

        # upper left panel (activation)
        plt.subplot2grid((6, 4), (0, 0), rowspan=3, colspan=2)
        self.activation(show=False, **kwargs)

        # lower left panel (trials)
        plt.subplot2grid((6, 4), (3, 0), rowspan=3, colspan=2)
        self.trials(show=False, **kwargs)

        # upper right (left) panel (PDF)
        plt.subplot2grid((6, 4), (0, 2), rowspan=2)
        self.pdf(show=False, **kwargs)

        # upper right (right) panel (CDF)
        plt.subplot2grid((6, 4), (0, 3), rowspan=2)
        self.cdf(show=False, **kwargs)

        # middle right panel (CAF)
        plt.subplot2grid((6, 4), (2, 2), rowspan=2, colspan=2)
        self.caf(show=False, **kwargs)

        # bottom right panel (delta)
        plt.subplot2grid((6, 4), (4, 2), rowspan=2, colspan=2)
        self.delta(show=False, **kwargs)

        plt.subplots_adjust(hspace=kwargs["hspace"], wspace=kwargs["wspace"])
        plt.show(block=False)

    def _summary2_res_th(self, **kwargs):

        # upper right (left) panel (PDF)
        plt.subplot2grid((3, 2), (0, 0))
        self.pdf(show=False, **kwargs)

        # upper right (eight) panel (CDF)
        plt.subplot2grid((3, 2), (0, 1))
        self.cdf(show=False, **kwargs)

        # middle left panel
        plt.subplot2grid((3, 2), (1, 0), colspan=2)
        self.caf(show=False, **kwargs)

        # bottom right panel
        plt.subplot2grid((3, 2), (2, 0), colspan=2)
        self.delta(show=False, **kwargs)

        plt.subplots_adjust(hspace=kwargs["hspace"], wspace=kwargs["wspace"])
        plt.show(block=False)

    def _summary3_res_th(self, **kwargs):

        # upper right (left) panel (PDF)
        plt.subplot2grid((3, 1), (0, 0))
        self.rt_correct(show=False, **kwargs)

        # upper right (eight) panel (CDF)
        plt.subplot2grid((3, 1), (1, 0))
        self.er(show=False, **kwargs)

        # middle left panel
        plt.subplot2grid((3, 1), (2, 0))
        self.rt_error(show=False, **kwargs)

        plt.subplots_adjust(hspace=kwargs["hspace"], wspace=kwargs["wspace"])
        plt.show(block=False)

    def _summary1_res_ob(self, **kwargs):
        """Plot summaty observed data."""

        # upper left panel (rt correct)
        plt.subplot2grid((3, 2), (0, 0))
        self.rt_correct(show=False, **kwargs)

        # middle left panel
        plt.subplot2grid((3, 2), (1, 0))
        self.er(show=False, **kwargs)

        # bottom left panel
        plt.subplot2grid((3, 2), (2, 0))
        self.rt_error(show=False, **kwargs)

        # upper right panel (cdf)
        plt.subplot2grid((3, 2), (0, 1))
        self.cdf(show=False, **kwargs)

        # middle right (left) panel (PDF)
        plt.subplot2grid((3, 2), (1, 1))
        self.caf(show=False, **kwargs)

        # lower right (right) panel (CDF)
        plt.subplot2grid((3, 2), (2, 1))
        self.delta(show=False, **kwargs)

        plt.subplots_adjust(hspace=kwargs["hspace"], wspace=kwargs["wspace"])
        plt.show(block=False)

    def activation(
        self,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_position: str = "best",
        colors: tuple = ("green", "red"),
        **kwargs,
    ):
        """Plot activation."""

        if not isinstance(self.res, Sim):
            print("Observed data does not have activation function!")
            return

        if not self.res.xt:
            print("Plotting activation function requires full_data=True")
            return

        if show:
            plt.figure(len(plt.get_fignums()) + 1)

        l_kws = _filter_dict(kwargs, plt.Line2D)

        plt.plot(self.res.eq4, "k-")
        plt.plot(self.res.eq4 * -1, "k--")
        plt.plot(self.res.xt[0], color=colors[0], label=cond_labels[0], **l_kws)
        plt.plot(self.res.xt[1], color=colors[1], label=cond_labels[1], **l_kws)
        plt.plot(
            np.cumsum(np.repeat(self.res.prms.drc, self.res.prms.t_max)),
            color="black",
            **l_kws,
        )
        self._bounds()

        kwargs.setdefault("xlim", [0, self.res.prms.t_max])
        kwargs.setdefault("ylim", [-self.res.prms.bnds - 20, self.res.prms.bnds + 20])
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", "E[X(t)]")

        _adjust_plt(**kwargs)

        if legend_position is not None:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def trials(
        self,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_position: str = "upper right",
        colors: tuple = ("green", "red"),
        **kwargs,
    ):
        """Plot individual trials."""

        if not isinstance(self.res, Sim):
            print("Observed data does not have individual trials!")
            return

        if not self.res.xt:
            print("Plotting individual trials function requires full_data=True")
            return

        if show:
            plt.figure(len(plt.get_fignums()) + 1)

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for trl in range(self.res.n_trls_data):
            if trl == 0:
                labels = cond_labels
            else:
                labels = [None, None]
            for comp in (0, 1):
                idx = np.where(
                    np.abs(self.res.data_trials[comp][trl, :]) >= self.res.prms.bnds
                )[0][0]
                plt.plot(
                    self.res.data_trials[comp][trl][0:idx],
                    color=colors[comp],
                    label=labels[comp],
                    **l_kws,
                )
        self._bounds()

        kwargs.setdefault("xlim", [0, self.res.prms.t_max])
        kwargs.setdefault("ylim", [-self.res.prms.bnds - 20, self.res.prms.bnds + 20])
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", "X(t)")

        _adjust_plt(**kwargs)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def _bounds(self):
        plt.axhline(y=self.res.prms.bnds, color="grey", linestyle="--")
        plt.axhline(y=-self.res.prms.bnds, color="grey", linestyle="--")

    def pdf(
        self,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_position: str = "upper right",
        colors: tuple = ("green", "red"),
        **kwargs,
    ):
        """Plot PDF."""

        if show:
            plt.figure(len(plt.get_fignums()) + 1)

        l_kws = _filter_dict(kwargs, plt.Line2D)

        if isinstance(self.res, Sim):

            for comp in (0, 1):
                pdf, axes = fastKDE.pdf(self.res.data[comp][0])
                plt.plot(
                    axes, pdf, color=colors[comp], label=cond_labels[comp], **l_kws
                )
            kwargs.setdefault("xlim", [0, self.res.prms.t_max])

        elif isinstance(self.res, Ob):

            for idx, comp in enumerate(("comp", "incomp")):
                data = np.array(self.res.data["RT"][self.res.data.Comp == comp])
                pdf, axes = fastKDE.pdf(data)
                plt.plot(axes, pdf, color=colors[idx], label=cond_labels[idx], **l_kws)
            kwargs.setdefault("xlim", [min(data) - 100, max(data) + 100])

        kwargs.setdefault("ylim", [0, 0.01])
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", "PDF")

        _adjust_plt(**kwargs)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def cdf(
        self,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_position: tuple = "lower right",
        colors: tuple = ("green", "red"),
        **kwargs,
    ):
        """Plot CDF."""

        if show:
            plt.figure(len(plt.get_fignums()) + 1)

        l_kws = _filter_dict(kwargs, plt.Line2D)
        if hasattr(self.res, "prms"):
            for comp in (0, 1):
                pdf, axes = fastKDE.pdf(self.res.data[comp][0])
                plt.plot(
                    axes,
                    np.cumsum(pdf) * np.diff(axes)[0:1],
                    color=colors[comp],
                    label=cond_labels[comp],
                    **l_kws,
                )

            kwargs.setdefault("xlim", [0, self.res.prms.t_max])

        if not hasattr(self.res, "prms"):

            print("here")
            kwargs.setdefault("marker", "o")
            kwargs.setdefault("markersize", 4)

            l_kws = _filter_dict(kwargs, plt.Line2D)
            for idx, comp in enumerate(("mean_comp", "mean_incomp")):
                plt.plot(
                    self.res.delta[comp],
                    np.linspace(0, 1, self.res.n_delta + 2)[1:-1],
                    color=colors[idx],
                    label=cond_labels[idx],
                    **l_kws,
                )

            kwargs.setdefault(
                "xlim",
                [
                    np.min(self.res.delta.mean_bin) - 100,
                    np.max(self.res.delta.mean_bin) + 100,
                ],
            )

        kwargs.setdefault("ylim", [0, 1.05])
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", "CDF")

        plt.axhline(y=1, color="grey", linestyle="--")
        _adjust_plt(**kwargs)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def caf(
        self,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_position: str = "lower right",
        colors: tuple = ("green", "red"),
        **kwargs,
    ):
        """Plot CAF."""

        if show:
            plt.figure(len(plt.get_fignums()) + 1)

        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, comp in enumerate(("comp", "incomp")):
            plt.plot(
                self.res.caf["bin"],
                self.res.caf[comp],
                color=colors[idx],
                label=cond_labels[idx],
                **l_kws,
            )

        plt.xticks(
            range(1, self.res.n_caf + 1), [str(x) for x in range(1, self.res.n_caf + 1)]
        )

        kwargs.setdefault("ylim", [0, 1.1])
        kwargs.setdefault("xlabel", "RT Bin")
        kwargs.setdefault("ylabel", "CAF")

        _adjust_plt(**kwargs)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def delta(self, show: bool = True, **kwargs):
        """Plot reaction-time delta plots."""

        if show:
            plt.figure(len(plt.get_fignums()) + 1)

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        datx, daty = self.res.delta["mean_bin"], self.res.delta["mean_effect"]

        l_kws = _filter_dict(kwargs, plt.Line2D)
        plt.plot(datx, daty, **l_kws)

        kwargs.setdefault("xlim", [np.min(datx) - 100, np.max(datx) + 100])
        kwargs.setdefault("ylim", [np.min(daty) - 25, np.max(daty) + 25])
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", r"$\Delta$  RT [ms]")

        _adjust_plt(**kwargs)

        if show:
            plt.show(block=False)

    def delta_errors(self, show: bool = True, **kwargs):
        """Plot error rate delta plots."""

        if show:
            plt.figure(len(plt.get_fignums()) + 1)

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        datx, daty = self.res.caf["bin"], self.res.caf["effect"]

        l_kws = _filter_dict(kwargs, plt.Line2D)
        plt.plot(datx, daty, **l_kws)

        plt.xticks(
            range(1, self.res.n_caf + 1), [str(x) for x in range(1, self.res.n_caf + 1)]
        )

        kwargs.setdefault("ylim", [np.min(daty) - 5, np.max(daty) + 5])
        kwargs.setdefault("xlabel", "RT Bin")
        kwargs.setdefault("ylabel", r"$\Delta$  ER [%]")

        _adjust_plt(**kwargs)

        if show:
            plt.show(block=False)

    def rt_correct(
        self,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        **kwargs,
    ):
        """Plot correct RT's."""

        if show:
            plt.figure(len(plt.get_fignums()) + 1)

        kwargs.setdefault("ylabel", "RT Correct [ms]")

        _plot_beh(
            self.res.summary["rt_cor"],
            cond_labels,
            False,
            **kwargs,
        )

        if show:
            plt.show(block=False)

    def er(
        self,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        **kwargs,
    ):
        """Plot error rate"""

        if show:
            plt.figure(len(plt.get_fignums()) + 1)

        kwargs.setdefault("ylabel", "Error Rate [%]")

        _plot_beh(
            self.res.summary["per_err"],
            cond_labels,
            True,
            **kwargs,
        )

        if show:
            plt.show(block=False)

    def rt_error(
        self,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        **kwargs,
    ):
        """Plot error RT's."""

        if show:
            plt.figure(len(plt.get_fignums()) + 1)

        kwargs.setdefault("ylabel", "RT Error [ms]")

        _plot_beh(
            self.res.summary["rt_err"],
            cond_labels,
            False,
            **kwargs,
        )

        if show:
            plt.show(block=False)


class PlotFit:
    def __init__(self, res: Fit):

        assert isinstance(res, Fit), "res must be of type 'Fit'"

        self.res_th = res.res_th
        self.res_ob = res.res_ob

    def summary(self, **kwargs):
        """Plot."""
        kwargs.setdefault("fig_type", "summary1")
        kwargs.setdefault("hspace", 0.5)
        kwargs.setdefault("wspace", 0.5)

        # upper left panel (rt correct)
        plt.subplot2grid((3, 2), (0, 0))
        self.rt_correct(show=False, **kwargs)

        # middle left panel
        plt.subplot2grid((3, 2), (1, 0))
        self.er(show=False, **kwargs)

        # bottom left panel
        plt.subplot2grid((3, 2), (2, 0))
        self.rt_error(show=False, **kwargs)

        # upper right panel (CDF)
        plt.subplot2grid((3, 2), (0, 1))
        self.cdf(show=False, **kwargs)

        # middle right panel (CAF)
        plt.subplot2grid((3, 2), (1, 1))
        self.caf(show=False, **kwargs)

        # bottom right panel (delta)
        plt.subplot2grid((3, 2), (2, 1))
        self.delta(show=False, **kwargs)

        plt.subplots_adjust(hspace=kwargs["hspace"], wspace=kwargs["wspace"])
        plt.show(block=False)

    def rt_correct(
        self,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_labels: tuple = ("Observed", "Predicted"),
        legend_position: str = "upper left",
        **kwargs,
    ):
        """Plot correct RT's."""

        kwargs.setdefault("ylabel", "RT Correct [ms]")
        kwargs.setdefault("colors", ("black", "grey")),
        kwargs.setdefault("linestyles", ("-", "--")),
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, res in enumerate((self.res_ob, self.res_th)):
            plt.plot(
                cond_labels,
                res.summary["rt_cor"],
                color=kwargs["colors"][idx],
                linestyle=kwargs["linestyles"][idx],
                label=legend_labels[idx],
                **l_kws,
            )

        kwargs.setdefault(
            "ylim",
            [
                np.min(self.res_ob.summary["rt_cor"]) - 100,
                np.max(self.res_ob.summary["rt_cor"]) + 100,
            ],
        )

        plt.margins(x=0.5)
        _adjust_plt(**kwargs)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def er(
        self,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_labels: tuple = ("Observed", "Predicted"),
        legend_position: str = "upper left",
        **kwargs,
    ):
        """Plot error rate."""

        kwargs.setdefault("ylabel", "Error Rate [%]")
        kwargs.setdefault("colors", ("black", "grey")),
        kwargs.setdefault("linestyles", ("-", "--")),
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, res in enumerate((self.res_ob, self.res_th)):
            plt.plot(
                cond_labels,
                res.summary["per_err"],
                color=kwargs["colors"][idx],
                linestyle=kwargs["linestyles"][idx],
                label=legend_labels[idx],
                **l_kws,
            )

        kwargs.setdefault(
            "ylim",
            [
                0,
                np.max(self.res_ob.summary["per_err"]) + 5,
            ],
        )

        plt.margins(x=0.5)
        _adjust_plt(**kwargs)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def rt_error(
        self,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_labels: tuple = ("Observed", "Predicted"),
        legend_position: str = "upper left",
        **kwargs,
    ):
        """Plot error RT's."""

        kwargs.setdefault("ylabel", "RT Errors [ms]")
        kwargs.setdefault("colors", ("black", "grey")),
        kwargs.setdefault("linestyles", ("-", "--")),
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, res in enumerate((self.res_ob, self.res_th)):
            plt.plot(
                cond_labels,
                res.summary["rt_cor"],
                color=kwargs["colors"][idx],
                linestyle=kwargs["linestyles"][idx],
                label=legend_labels[idx],
                **l_kws,
            )

        kwargs.setdefault(
            "ylim",
            [
                np.min(self.res_ob.summary["rt_err"]) - 100,
                np.max(self.res_ob.summary["rt_err"]) + 100,
            ],
        )

        plt.margins(x=0.5)
        _adjust_plt(**kwargs)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def cdf(
        self,
        show: bool = True,
        legend_labels: tuple = (
            "Compatible Observed",
            "Incompatible Observed",
            "Compatible Predicted",
            "Incompatible Predicted",
        ),
        legend_position: str = "lower right",
        colors: tuple = ("green", "red"),
        **kwargs,
    ):
        """Plot CDF."""

        kwargs.setdefault("linestyle", "None")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)
        kwargs.setdefault("ylabel", "CDF")

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, comp in enumerate(("mean_comp", "mean_incomp")):
            plt.plot(
                self.res_ob.delta[comp],
                np.linspace(0, 1, self.res_ob.n_delta + 2)[1:-1],
                color=colors[idx],
                label=legend_labels[idx],
                **l_kws,
            )

        kwargs["linestyle"] = "-"
        kwargs["marker"] = "None"

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, comp in enumerate(("mean_comp", "mean_incomp")):
            plt.plot(
                self.res_th.delta[comp],
                np.linspace(0, 1, self.res_th.n_delta + 2)[1:-1],
                color=colors[idx],
                label=legend_labels[idx + 2],
                **l_kws,
            )

        kwargs.setdefault(
            "xlim",
            [
                np.min(self.res_ob.delta.mean_bin) - 100,
                np.max(self.res_ob.delta.mean_bin) + 100,
            ],
        )

        plt.margins(x=0.5)
        _adjust_plt(**kwargs)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def caf(
        self,
        show: bool = True,
        legend_labels: tuple = (
            "Compatible Observed",
            "Incompatible Observed",
            "Compatible Predicted",
            "Incompatible Predicted",
        ),
        legend_position: str = "lower right",
        colors: tuple = ("green", "red"),
        **kwargs,
    ):
        """Plot CAF."""

        kwargs.setdefault("linestyle", "None")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)
        kwargs.setdefault("xlabel", "RT Bin")
        kwargs.setdefault("ylabel", "CAF")

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, comp in enumerate(("comp", "incomp")):
            plt.plot(
                self.res_ob.caf["bin"],
                self.res_ob.caf[comp],
                color=colors[idx],
                label=legend_labels[idx],
                **l_kws,
            )

        kwargs["linestyle"] = "-"
        kwargs["marker"] = "None"
        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, comp in enumerate(("comp", "incomp")):
            plt.plot(
                self.res_th.caf["bin"],
                self.res_th.caf[comp],
                color=colors[idx],
                label=legend_labels[idx + 2],
                **l_kws,
            )

        kwargs.setdefault("ylim", (0, 1.01))

        plt.xticks(
            range(1, self.res_th.n_caf + 1),
            [str(x) for x in range(1, self.res_th.n_caf + 1)],
        )
        _adjust_plt(**kwargs)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def delta(
        self,
        show: bool = True,
        legend_labels: tuple = ("Observed", "Predicted"),
        legend_position: str = "lower right",
        **kwargs,
    ):
        """Plot reaction-time delta plots."""
        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)
        kwargs.setdefault("linestyle", "None")
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", r"$\Delta$ RT [ms]")

        l_kws = _filter_dict(kwargs, plt.Line2D)
        plt.plot(
            self.res_ob.delta["mean_bin"],
            self.res_ob.delta["mean_effect"],
            label=legend_labels[0],
            **l_kws,
        )

        kwargs["linestyle"] = "-"
        kwargs["marker"] = "None"
        l_kws = _filter_dict(kwargs, plt.Line2D)
        plt.plot(
            self.res_th.delta["mean_bin"],
            self.res_th.delta["mean_effect"],
            label=legend_labels[1],
            **l_kws,
        )

        kwargs.setdefault(
            "xlim",
            [
                np.min(self.res_ob.delta.mean_bin) - 100,
                np.max(self.res_ob.delta.mean_bin) + 100,
            ],
        )
        kwargs.setdefault(
            "ylim",
            [
                np.min(self.res_ob.delta.mean_effect) - 25,
                np.max(self.res_ob.delta.mean_effect) + 25,
            ],
        )

        _adjust_plt(**kwargs)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def delta_errors(
        self,
        show: bool = True,
        legend_labels: tuple = ("Observed", "Predicted"),
        legend_position: str = "upper right",
        **kwargs,
    ):
        """Plot error-rate delta plots."""
        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)
        kwargs.setdefault("linestyle", "None")
        kwargs.setdefault("xlabel", "RT Bin")
        kwargs.setdefault("ylabel", r"$\Delta$  ER [%]")

        l_kws = _filter_dict(kwargs, plt.Line2D)
        plt.plot(
            self.res_ob.caf["bin"],
            self.res_ob.caf["effect"],
            label=legend_labels[0],
            **l_kws,
        )

        kwargs["linestyle"] = "-"
        kwargs["marker"] = "None"
        l_kws = _filter_dict(kwargs, plt.Line2D)
        plt.plot(
            self.res_th.caf["bin"],
            self.res_th.caf["effect"],
            label=legend_labels[1],
            **l_kws,
        )

        kwargs.setdefault(
            "ylim",
            [
                np.min(self.res_ob.caf.effect) - 5,
                np.max(self.res_ob.caf.effect) + 5,
            ],
        )

        plt.xticks(
            range(1, self.res_th.n_caf + 1),
            [str(x) for x in range(1, self.res_th.n_caf + 1)],
        )
        _adjust_plt(**kwargs)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)


def _plot_beh(dat, cond_labels: tuple, zeroed: bool, **kwargs):
    """Internal function to plot rt/er for comp vs. comp"""

    kwargs.setdefault("color", "black")
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", 4)

    l_kws = _filter_dict(kwargs, plt.Line2D)
    plt.plot(cond_labels, dat, **l_kws)
    plt.margins(x=0.5)

    if zeroed:
        kwargs.setdefault("ylim", [0, np.max(dat) + 5])
    else:
        kwargs.setdefault("ylim", [np.min(dat) - 100, np.max(dat) + 100])

    _adjust_plt(**kwargs)


def _filter_dict(given, allowed):
    """Internal function to filter dict **kwargs for allowd arguments."""
    f = {}
    allowed = inspect.signature(allowed).parameters
    for k, v in given.items():
        if k in allowed.keys():
            f[k] = v
    return f


def _adjust_plt(**kwargs):
    """Internal function to adjust some common plot properties."""
    plt.xlim(kwargs.get("xlim", None))
    plt.ylim(kwargs.get("ylim", None))
    plt.xlabel(kwargs.get("xlabel", ""), fontsize=kwargs.get("label_fontsize", 12))
    plt.ylabel(kwargs.get("ylabel", ""), fontsize=kwargs.get("label_fontsize", 12))
    plt.xticks(fontsize=kwargs.get("tick_fontsize", 10))
    plt.yticks(fontsize=kwargs.get("tick_fontsize", 10))
