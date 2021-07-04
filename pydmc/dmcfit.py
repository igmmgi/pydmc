import numpy as np
from scipy.optimize import fmin
from pydmc.dmcsim import DmcSim
import matplotlib.pyplot as plt


class DmcFit:
    def __init__(
        self,
        res_ob,
        n_trls=100000,
        start_vals=None,
        min_vals=None,
        max_vals=None,
        fixed_fit=None,
        n_delta=19,
        n_caf=5,
        var_sp=True,
    ):
        """
        Parameters
        ----------
        """
        self.res_ob = res_ob
        self.res_th = None
        self.n_trls = n_trls
        self.start_vals = start_vals
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.fixed_fit = fixed_fit
        self.n_delta = n_delta
        self.n_caf = n_caf
        self.var_sp = var_sp
        self.cost_value = np.Inf
        start_vals = {
            "amp": 20,
            "tau": 100,
            "drc": 0.2,
            "bnds": 75,
            "res_mean": 300,
            "res_sd": 30,
            "aa_shape": 2,
            "sp_shape": 3,
            "sigma": 4,
        }
        if self.start_vals is None:
            self.start_vals = start_vals
        else:
            start_vals.update(self.start_vals)
            self.start_vals = start_vals
        min_vals = {
            "amp": 0,
            "tau": 5,
            "drc": 0.1,
            "bnds": 20,
            "res_mean": 200,
            "res_sd": 5,
            "aa_shape": 1,
            "sp_shape": 2,
            "sigma": 1,
        }
        if self.min_vals is None:
            self.min_vals = min_vals
        else:
            min_vals.update(self.min_vals)
            self.min_vals = min_vals
        max_vals = {
            "amp": 40,
            "tau": 300,
            "drc": 1.0,
            "bnds": 150,
            "res_mean": 800,
            "res_sd": 100,
            "aa_shape": 3,
            "sp_shape": 4,
            "sigma": 10,
        }
        if self.max_vals is None:
            self.max_vals = max_vals
        else:
            max_vals.update(self.max_vals)
            self.max_vals = max_vals
        fixed_fit = {
            "amp": False,
            "tau": False,
            "drc": False,
            "bnds": False,
            "res_mean": False,
            "res_sd": False,
            "aa_shape": False,
            "sp_shape": False,
            "sigma": True,
        }
        if self.fixed_fit is None:
            self.fixed_fit = fixed_fit
        else:
            fixed_fit.update(self.fixed_fit)
            self.fixed_fit = fixed_fit

        for key, value in self.fixed_fit.items():
            if value:
                self.min_vals[key] = self.start_vals[key]
                self.max_vals[key] = self.start_vals[key]

    def fit_data(self, **kwargs):
        self.fit = fmin(
            self._function_to_minimise,
            np.array(list(self.start_vals.values())),
            (self.res_ob,),
            **kwargs,
        )

    def _function_to_minimise(self, x, res_ob):

        # bounds hack
        x = np.maximum(x, list(self.min_vals.values()))
        x = np.minimum(x, list(self.max_vals.values()))

        self.res_th = DmcSim(
            amp=x[0],
            tau=x[1],
            drc=x[2],
            bnds=x[3],
            res_mean=x[4],
            res_sd=x[5],
            aa_shape=x[6],
            sp_shape=x[7],
            sigma=x[8],
            var_sp=True,
            n_trls=self.n_trls,
            n_delta=self.n_delta,
            n_caf=self.n_caf,
            res_dist=1,
        )

        print(
            f"amp:{x[0]:4.1f}",
            f"tau:{x[1]:4.1f}",
            f"drc:{x[2]:4.2f}",
            f"bnds:{x[3]:4.1f}",
            f"res_mean:{x[4]:4.0f}",
            f"res_sd:{x[5]:4.1f}",
            f"aa_shape:{x[6]:4.1f}",
            f"sp_shape={x[7]:4.1f}",
            f"| cost={self.cost_value:.2f}",
        )

        self.cost_value = DmcFit.calculate_cost_value_rmse(self.res_th, res_ob)

        return self.cost_value

    @staticmethod
    def calculate_cost_value_rmse(res_th, res_ob):
        """calculate_cost_value_rmse"""
        n_rt = len(res_th.delta) * 2
        n_err = len(res_th.caf)

        np.sum(
            np.sum(
                res_th.delta[["mean_comp", "mean_incomp"]]
                - res_ob.delta[["mean_comp", "mean_incomp"]]
            )
        )

        cost_caf = np.sqrt(
            (1 / n_err) * np.sum((res_th.caf["Error"] - res_ob.caf["Error"]) ** 2)
        )

        cost_rt = np.sqrt(
            (1 / n_rt)
            * np.sum(
                np.sum(
                    res_th.delta[["mean_comp", "mean_incomp"]]
                    - res_ob.delta[["mean_comp", "mean_incomp"]]
                )
            )
        )

        weight_rt = n_rt / (n_rt + n_err)
        weight_caf = (1 - weight_rt) * 1500

        cost_value = (weight_caf * cost_caf) + (weight_rt * cost_rt)

        if np.isnan(cost_value):
            cost_value = np.inf

        return cost_value

    def plot(self):
        """Plot."""

        # upper left panel (rt correct)
        plt.subplot2grid((3, 1), (0, 0))
        self.plot_rt_correct(show=False)

        # middle left pannel
        plt.subplot2grid((3, 1), (1, 0))
        self.plot_er(show=False)

        # bottom left pannel
        plt.subplot2grid((3, 1), (2, 0))
        self.plot_rt_error(show=False)

    def plot_rt_correct(
        self,
        show=True,
        ylim=None,
        xlabel=None,
        ylabel="RT Correct [ms]",
        label_fontsize=12,
        tick_fontsize=10,
        cond_labels=("Compatible", "Incompatible"),
        colors=("black", "grey"),
        **kwargs,
    ):
        """Plot correct RT's."""
        if ylim is None:
            ylim = [
                min(
                    min(self.res_th.summary["rt_cor"]),
                    min(self.res_ob.summary["rt_cor"]),
                ),
                max(
                    max(self.res_th.summary["rt_cor"]),
                    max(self.res_ob.summary["rt_cor"]),
                ),
            ]
        self.res_th.plot_rt_correct(
            ylim=ylim,
            xlabel=xlabel,
            ylabel=ylabel,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            cond_labels=cond_labels,
            color=colors[0],
            **kwargs,
        )
        self.res_ob.plot_rt_correct(
            ylim=ylim,
            xlabel=xlabel,
            ylabel=ylabel,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            cond_labels=cond_labels,
            color=colors[1],
            **kwargs,
        )

        if show:
            plt.show(block=False)

    def plot_er(
        self,
        show=True,
        ylim=None,
        xlabel=None,
        ylabel="RT Correct [ms]",
        label_fontsize=12,
        tick_fontsize=10,
        cond_labels=("Compatible", "Incompatible"),
        colors=("black", "grey"),
    ):
        """Plot error rate."""
        self.res_ob.plot_er(
            color=colors[0],
            ylim=ylim,
            xlabel=xlabel,
            ylable=ylabel,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            cond_labels=cond_labels,
        )
        self.res_th.plot_er(
            color=colors[1],
            ylim=ylim,
            xlabel=xlabel,
            ylable=ylabel,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            cond_labels=cond_labels,
        )

        if show:
            plt.show(block=False)

    def plot_rt_error(
        self,
        show=True,
        ylim=None,
        xlabel=None,
        ylabel="RT Correct [ms]",
        label_fontsize=12,
        tick_fontsize=10,
        cond_labels=("Compatible", "Incompatible"),
        colors=("black", "grey"),
    ):
        """Plot error RT's."""
        self.res_ob.plot_rt_error(
            ylim=ylim,
            color=colors[0],
            xlabel=xlabel,
            ylable=ylabel,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            cond_labels=cond_labels,
        )
        self.res_th.plot_rt_error(
            ylim=ylim,
            color=colors[1],
            xlabel=xlabel,
            ylable=ylabel,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            cond_labels=cond_labels,
        )

        if show:
            plt.show(block=False)

    def plot_cdf(
        self,
        show=True,
        ylim=(0, 1.05),
        xlabel=None,
        cond_labels=("Compatible", "Incompatible"),
        ylabel="RT Error [ms]",
        label_fontsize=12,
        tick_fontsize=10,
        colors=("green", "grey"),
    ):

        self.res_ob.plot_cdf(
            ylim=ylim,
            color=colors[0],
            xlabel=xlabel,
            ylable=ylabel,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            cond_labels=cond_labels,
        )
        self.res_th.plot_cdf(
            ylim=ylim,
            color=colors[1],
            xlabel=xlabel,
            ylable=ylabel,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            cond_labels=cond_labels,
        )

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
        colors=("green", "red"),
        **kwargs,
    ):

        self.res_ob.plot_caf(
            ylim=ylim,
            color=colors[0],
            xlabel=xlabel,
            ylable=ylabel,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            cond_labels=cond_labels,
            **kwargs,
        )
        self.res_th.plot_caf(
            ylim=ylim,
            color=colors[1],
            xlabel=xlabel,
            ylable=ylabel,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            cond_labels=cond_labels,
            **kwargs,
        )
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
        **kwargs,
    ):
        """Plot reaction-time delta plots."""

        self.res_ob.plot_delta(
            ylim=ylim,
            xlim=xlim,
            xlabel=xlabel,
            ylable=ylabel,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )
        self.res_th.plot_delta(
            ylim=ylim,
            xlim=xlim,
            xlabel=xlabel,
            ylable=ylabel,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )
        if show:
            plt.show(block=False)
