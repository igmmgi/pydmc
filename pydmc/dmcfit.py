import numpy as np
import pandas as pd
from copy import deepcopy
from dataclasses import dataclass, asdict
from scipy.optimize import minimize, differential_evolution
from pydmc.dmcsim import Sim, Prms
from pydmc.dmcob import Ob


@dataclass
class PrmsFit:
    # start, min, max, fitted
    amp: tuple = (20, 0, 40, True)
    tau: tuple = (30, 5, 300, True)
    drc: tuple = (0.5, 0.1, 1.0, True)
    bnds: tuple = (75, 20, 150, True)
    res_mean: tuple = (300, 200, 800, True)
    res_sd: tuple = (30, 5, 100, True)
    aa_shape: tuple = (2, 1, 3, True)
    sp_shape: tuple = (3, 2, 4, True)
    sigma: tuple = (4, 1, 10, False)

    def set_start_values(self, **kwargs) -> None:
        [setattr(self, k, (v,) + getattr(self, k)[1:]) for k, v in kwargs.items()]

    def set_min_values(self, **kwargs) -> None:
        [
            setattr(self, k, (getattr(self, k)[0],) + (v,) + getattr(self, k)[2:])
            for k, v in kwargs.items()
        ]

    def set_max_values(self, **kwargs) -> None:
        [
            setattr(self, k, getattr(self, k)[0:2] + (v,) + (getattr(self, k)[3],))
            for k, v in kwargs.items()
        ]

    def set_fitted_values(self, **kwargs) -> None:
        [setattr(self, k, getattr(self, k)[0:3] + (v,)) for k, v in kwargs.items()]

    def dmc_prms(self, sp_dist: int = 1) -> Prms:
        return Prms(
            **self.dict(0), sp_dist=sp_dist, sp_lim=(-self.bnds[0], self.bnds[1])
        )

    def array(self, idx: int = 0) -> list:
        return [x[idx] for x in asdict(self).values() if x[-1]]

    def dict(self, idx: int = 0) -> dict:
        return {k: v[idx] for k, v in asdict(self).items() if v}

    def bounds(self) -> list:
        return [x[1:3] for x in asdict(self).values() if x[-1]]


class Fit:
    def __init__(
        self,
        res_ob: Ob,
        n_trls: int = 100_000,
        start_vals: PrmsFit = PrmsFit(),
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
        self.dmc_prms = start_vals.dmc_prms()
        self.n_delta = n_delta
        self.p_delta = p_delta
        self.t_delta = t_delta
        self.n_caf = n_caf
        self.cost_function = self._assign_cost_function(cost_function)
        self.cost_value = np.Inf

    def _assign_cost_function(self, cost_function: str):
        if cost_function == "RMSE":
            return self.calculate_cost_value_rmse
        elif cost_function == "SPE":
            return self.calculate_cost_value_spe
        else:
            raise Exception("cost function not implemented!")

    def _fit_initial_grid(self):
        pass

    def fit_data_neldermead(self, **kwargs) -> None:
        self.res_th = Sim(deepcopy(self.dmc_prms))
        kwargs.setdefault("maxiter", 500)
        self.fit = minimize(
            self._function_to_minimise,
            np.array(self.start_vals.array(0)),
            method="nelder-mead",
            bounds=self.start_vals.bounds(),
            options=kwargs,
        )

    def fit_data_differential_evolution(self, **kwargs) -> None:
        self.res_th = Sim(deepcopy(self.start_vals.dmc_prms()))
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
            if getattr(self.start_vals, k)[-1]:
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


class FitSubjects:
    def __init__(self, res_ob: Ob):
        self.res_ob = res_ob
        self.subjects = np.unique(res_ob.summary_subject.Subject)
        self.fits = self._split_subjects()

    def _split_subjects(self) -> list[Fit]:
        return [deepcopy(Fit(self.res_ob.select_subject(s))) for s in self.subjects]

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
