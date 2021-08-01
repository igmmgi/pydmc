import numpy as np
import pandas as pd
from copy import deepcopy
from dataclasses import dataclass, fields, astuple
from scipy.optimize import minimize, differential_evolution
from pydmc.dmcsim import Sim, Prms
from pydmc.dmcplot import PlotFit


@dataclass
class PrmsBounds:
    amp: tuple = (0, 40)
    tau: tuple = (5, 300)
    drc: tuple = (0.1, 1.0)
    bnds: tuple = (20, 150)
    res_mean: tuple = (200, 800)
    res_sd: tuple = (5, 100)
    aa_shape: tuple = (1, 3)
    sp_shape: tuple = (2, 4)
    sigma: tuple = (4, 4)


class Fit:
    def __init__(
        self,
        res_ob,
        n_trls=100000,
        start_vals=Prms(sp_dist=1),
        bound_vals=PrmsBounds(),
        n_delta=19,
        p_delta=None,
        t_delta=1,
        n_caf=5,
        cost_function="RMSE",
        sp_dist=1,
    ):
        self.res_ob = res_ob
        self.res_th = None
        self.fit = None
        self.n_trls = n_trls
        self.start_vals = start_vals
        self.bound_vals = bound_vals
        self._min_vals = self._fieldvalues(0)
        self._max_vals = self._fieldvalues(1)
        self.n_delta = n_delta
        self.p_delta = p_delta
        self.t_delta = t_delta
        self.n_caf = n_caf
        self.sp_dist = sp_dist
        self.cost_function = self._assign_cost_function(cost_function)
        self.cost_value = np.Inf

    def _fieldvalues(self, idx):
        return [getattr(self.bound_vals, f.name)[idx] for f in fields(self.bound_vals)]

    def _assign_cost_function(self, cost_function):
        if cost_function == "RMSE":
            return self.calculate_cost_value_rmse
        elif cost_function == "SPE":
            return self.calculate_cost_value_spe
        else:
            raise Exception("cost function not implemented!")

    def fit_data_neldermead(self, **kwargs):
        self.res_th = Sim(self.start_vals)
        kwargs.setdefault("maxiter", 500)
        self.fit = minimize(
            self._function_to_minimise,
            [getattr(self.start_vals, f.name) for f in fields(self.start_vals)][:9],
            method="Nelder-Mead",
            bounds=astuple(self.bound_vals),
            options=kwargs,
        )

    def fit_data_differential_evolution(self, **kwargs):
        self.res_th = Sim(self.start_vals)
        self.fit = differential_evolution(
            self._function_to_minimise,
            astuple(self.bound_vals),
            **kwargs,
        )

    def print_summary(self):
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

    def table_summary(self):
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

    def _function_to_minimise(self, x):

        self._update_parameters(x)
        self.res_th.run_simulation()
        self.cost_value = self.cost_function(self.res_th, self.res_ob)
        self.print_summary()

        return self.cost_value

    def _update_parameters(self, x):

        self.res_th.prms.amp = x[0]
        self.res_th.prms.tau = x[1]
        self.res_th.prms.drc = x[2]
        self.res_th.prms.bnds = x[3]
        self.res_th.prms.res_mean = x[4]
        self.res_th.prms.res_sd = x[5]
        self.res_th.prms.aa_shape = x[6]
        self.res_th.prms.sp_shape = x[7]
        self.res_th.prms.sigma = x[8]
        self.res_th.prms.sp_dist = 1
        self.res_th.prms.sp_lim = (-x[3], x[3])

    @staticmethod
    def calculate_cost_value_rmse(res_th, res_ob):
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
                np.sum(res_th.caf[["comp", "incomp"]] - res_ob.caf[["comp", "incomp"]])
                ** 2
            )
        )

        cost_rt = np.sqrt(
            (1 / n_rt)
            * np.sum(
                np.sum(
                    res_th.delta[["mean_comp", "mean_incomp"]]
                    - res_ob.delta[["mean_comp", "mean_incomp"]]
                )
                ** 2
            )
        )

        weight_rt = n_rt / (n_rt + n_err)
        weight_caf = (1 - weight_rt) * 1500

        return (weight_caf * cost_caf) + (weight_rt * cost_rt)

    @staticmethod
    def calculate_cost_value_spe(res_th, res_ob):
        """calculate_cost_calue_spe

        Calculate Squared Percentage Error between simulated
        and observed data points

        Parameters
        ---------
        res_th
        res_ob
        """
        cost_caf = np.sum(
            ((res_ob.caf["Error"] - res_th.caf["Error"]) / res_ob.caf["Error"]) ** 2
        )

        cost_rt = np.sum(
            (
                (res_ob.delta.iloc[:, 1:3] - res_th.delta.iloc[:, 1:3])
                / res_ob.delta.iloc[:, 1:3]
            )
            ** 2
        ).sum()

        return cost_rt + cost_caf

    def plot(self, **kwargs):
        """Plot."""
        PlotFit(self.res_th, self.res_ob).plot(**kwargs)

    def plot_rt_correct(self, **kwargs):
        """Plot reaction time correct."""
        PlotFit(self.res_th, self.res_ob).plot_rt_correct(**kwargs)

    def plot_er(self, **kwargs):
        """Plot erorr rate."""
        PlotFit(self.res_th, self.res_ob).plot_er(**kwargs)

    def plot_rt_error(self, **kwargs):
        """Plot reaction time errors."""
        PlotFit(self.res_th, self.res_ob).plot_rt_error(**kwargs)

    def plot_cdf(self, **kwargs):
        """Plot CDF."""
        PlotFit(self.res_th, self.res_ob).plot_cdf(**kwargs)

    def plot_caf(self, **kwargs):
        """Plot CAF."""
        PlotFit(self.res_th, self.res_ob).plot_caf(**kwargs)

    def plot_delta(self, **kwargs):
        """Plot delta."""
        PlotFit(self.res_th, self.res_ob).plot_delta(**kwargs)

    def plot_delta_errors(self, **kwargs):
        """Plot delta errors."""
        PlotFit(self.res_th, self.res_ob).plot_delta_errors(**kwargs)


class FitSubjects:
    def __init__(self, res_ob):
        self.res_ob = res_ob
        self.subjects = np.unique(res_ob.summary_subject.Subject)
        self.fits = self._split_subjects()

    def _split_subjects(self):
        return [deepcopy(Fit(self.res_ob.select_subject(s))) for s in self.subjects]

    def fit_data_neldermead(self, **kwargs):
        """Fit data using neldermead."""
        [f.fit_data_neldermead(**kwargs) for f in self.fits]

    def fit_data_differential_evolution(self, **kwargs):
        """Fit data using differential evolution."""
        [f.fit_data_differential_evolution(**kwargs) for f in self.fits]

    def print_summary(self):
        """Print summary of individual fits."""
        for idx, f in enumerate(self.fits):
            print(f"Subject: {idx+1}\t")
            f.print_summary()

    def table_summary(self):
        """Combine tables of individual fits."""
        tables = []
        for idx, f in enumerate(self.fits):
            tmp_table = f.table_summary()
            tmp_table.insert(0, "Subject", idx + 1)
            tables.append(tmp_table)
        return pd.concat(tables)
