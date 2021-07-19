import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from scipy.optimize import fmin
from pydmc.dmcsim import DmcSim, DmcParameters


@dataclass
class DmcParametersFixed:
    amp: False
    tau: False
    drc: False
    bnds: False
    res_mean: False
    res_sd: False
    aa_shape: False
    sp_shape: False
    sigma: True


class DmcFit:
    def __init__(
        self,
        res_ob,
        n_trls=100000,
        start_vals=DmcParameters(
            amp=20,
            tau=200,
            drc=0.5,
            bnds=75,
            res_mean=300,
            res_sd=30,
            aa_shape=2,
            sp_shape=3,
            sigma=4,
        ),
        min_vals=DmcParameters(
            amp=0,
            tau=5,
            drc=0.1,
            bnds=20,
            res_mean=200,
            res_sd=5,
            aa_shape=1,
            sp_shape=2,
            sigma=1,
        ),
        max_vals=DmcParameters(
            amp=40,
            tau=300,
            drc=1.0,
            bnds=150,
            res_mean=800,
            res_sd=100,
            aa_shape=3,
            sp_shape=4,
            sigma=10,
        ),
        fixed_fit=None,
        n_delta=19,
        p_delta=None,
        t_delta=1,
        n_caf=5,
        var_sp=True,
    ):
        """
        Parameters
        ----------
        res_ob
        n_trls
        start_vals
        min_vals
        max_vals
        fixed_fit
        n_delta
        p_delta
        t_delta
        n_caf
        var_sp
        """
        self.res_ob = res_ob
        self.res_th = DmcSim
        self.n_trls = n_trls
        self.start_vals = start_vals
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.fixed_fit = fixed_fit
        self.n_delta = n_delta
        self.p_delta = p_delta
        self.t_delta = t_delta
        self.n_caf = n_caf
        self.var_sp = var_sp
        self.cost_value = np.Inf
        self.fixed_fit = DmcParametersFixed()
        for key, value in self.fixed_fit.__dict__.items():
            if value:
                self.min_vals[key] = self.start_vals[key]
                self.max_vals[key] = self.start_vals[key]


    def fit_data(self, **kwargs):
        self.res_th = DmcSim(DmcParameters(**self.start_vals))
        self.fit = fmin(
            self._function_to_minimise,
            np.array(list(self.start_vals.values())),
            **kwargs,
        )

    def summary(self):
        """Print summary of DmcFit."""
        print(
            f"amp:{self.res_th.amp:4.1f}",
            f"tau:{self.res_th.tau:4.1f}",
            f"drc:{self.res_th.drc:4.2f}",
            f"bnds:{self.res_th.bnds:4.1f}",
            f"res_mean:{self.res_th.res_mean:4.0f}",
            f"res_sd:{self.res_th.res_sd:4.1f}",
            f"aa_shape:{self.res_th.aa_shape:4.1f}",
            f"sp_shape:{self.res_th.sp_shape:4.1f}",
            f"| cost={self.cost_value:.2f}",
        )

    def _function_to_minimise(self, x):

        # bounds hack
        x = np.maximum(x, list(self.min_vals.values()))
        x = np.minimum(x, list(self.max_vals.values()))

        self.res_th.prms.amp = x[0]
        self.res_th.prms.tau = x[1]
        self.res_th.prms.drc = x[2]
        self.res_th.prms.bnds = x[3]
        self.res_th.prms.res_mean = x[4]
        self.res_th.prms.res_sd = x[5]
        self.res_th.prms.aa_shape = x[6]
        self.res_th.prms.sp_shape = x[7]
        self.res_th.prms.sigma = x[8]
        self.res_th.prms.var_sp = True
        self.res_th.prms.sp_lim = (-x[3], x[3])

        self.res_th.run_simulation()
        self.cost_value = DmcFit.calculate_cost_value_rmse(self.res_th, self.res_ob)
        self.summary()

        return self.cost_value

    @staticmethod
    def calculate_cost_value_rmse(res_th, res_ob):
        """calculate_cost_value_rmse

        Parameters
        ----------
        res_th
        res_ob
        """
        n_rt = len(res_th.delta) * 2
        n_err = len(res_th.caf)

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
            ** 2
        )

        weight_rt = n_rt / (n_rt + n_err)
        weight_caf = (1 - weight_rt) * 1500

        cost_value = (weight_caf * cost_caf) + (weight_rt * cost_rt)

        return cost_value

    def plot(
        self, label_fontsize=12, tick_fontsize=10, hspace=0.5, wspace=0.5, **kwargs
    ):
        """Plot.

        Parameters
        ----------
        label_fontsize
        tick_fontsize
        hspace
        wspace
        kwargs
        """

        # upper left panel (rt correct)
        plt.subplot2grid((3, 2), (0, 0))
        self.plot_rt_correct(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # middle left panel
        plt.subplot2grid((3, 2), (1, 0))
        self.plot_er(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # bottom left panel
        plt.subplot2grid((3, 2), (2, 0))
        self.plot_rt_error(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # upper right panel (CDF)
        plt.subplot2grid((3, 2), (0, 1))
        self.plot_cdf(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # middle right panel (CAF)
        plt.subplot2grid((3, 2), (1, 1))
        self.plot_caf(
            show=False,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

        # bottom right panel (delta)
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
        ylabel="RT Correct [ms]",
        label_fontsize=12,
        tick_fontsize=10,
        cond_labels=("Compatible", "Incompatible"),
        colors=("black", "grey"),
        linestyles=("-", "--"),
        legend_labels=("Observed", "Predicted"),
        legend_position="upper left",
        **kwargs,
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
        colors
        linestyles
        legend_labels
        legend_position
        kwargs
        """

        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        plt.plot(
            cond_labels,
            self.res_ob.summary["rt_cor"],
            color=colors[0],
            linestyle=linestyles[0],
            label=legend_labels[0],
            **kwargs,
        )
        plt.plot(
            cond_labels,
            self.res_th.summary["rt_cor"],
            color=colors[1],
            linestyle=linestyles[1],
            label=legend_labels[1],
            **kwargs,
        )

        if ylim is None:
            ylim = [
                np.min(self.res_ob.summary["rt_cor"]) - 100,
                np.max(self.res_ob.summary["rt_cor"]) + 100,
            ]

        plt.margins(x=0.5)
        _adjust_plt(None, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if legend_position:
            plt.legend(loc=legend_position)

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
        colors=("black", "grey"),
        linestyles=("-", "--"),
        legend_labels=("Observed", "Predicted"),
        legend_position="upper left",
        **kwargs,
    ):
        """Plot error rate.

        Parameters
        ----------
        show
        ylim
        xlabel
        ylabel
        label_fontsize
        tick_fontsize
        cond_labels
        colors
        linestyles
        legend_labels
        legend_position
        kwargs
        """

        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        plt.plot(
            cond_labels,
            self.res_ob.summary["per_err"],
            color=colors[0],
            linestyle=linestyles[0],
            label=legend_labels[0],
            **kwargs,
        )
        plt.plot(
            cond_labels,
            self.res_th.summary["per_err"],
            color=colors[1],
            linestyle=linestyles[1],
            label=legend_labels[1],
            **kwargs,
        )

        if ylim is None:
            ylim = [0, np.max(self.res_ob.summary["per_err"]) + 5]

        plt.margins(x=0.5)
        _adjust_plt(None, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if legend_position:
            plt.legend(loc=legend_position)

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
        linestyles=("-", "--"),
        legend_labels=("Observed", "Predicted"),
        legend_position="upper left",
        **kwargs,
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
        colors
        linestyles
        legend_labels
        legend_position
        kwargs
        """

        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        plt.plot(
            cond_labels,
            self.res_ob.summary["rt_err"],
            color=colors[0],
            linestyle=linestyles[0],
            label=legend_labels[0],
            **kwargs,
        )
        plt.plot(
            cond_labels,
            self.res_th.summary["rt_err"],
            color=colors[1],
            linestyle=linestyles[1],
            label=legend_labels[1],
            **kwargs,
        )

        if ylim is None:
            ylim = [
                np.min(self.res_ob.summary["rt_err"]) - 100,
                np.max(self.res_ob.summary["rt_err"]) + 100,
            ]

        plt.margins(x=0.5)
        _adjust_plt(None, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def plot_cdf(
        self,
        show=True,
        xlim=None,
        xlabel=None,
        legend_labels=(
            "Compatible Observed",
            "Incompatible Observed",
            "Compatible Predicted",
            "Incompatible Predicted",
        ),
        legend_position="lower right",
        ylabel="CDF",
        label_fontsize=12,
        tick_fontsize=10,
        colors=("green", "red"),
        **kwargs,
    ):
        """

        Parameters
        ----------
        show
        xlim
        xlabel
        legend_labels
        legend_position
        ylabel
        label_fontsize
        tick_fontsize
        colors
        kwargs
        """
        kwargs.setdefault("linestyle", "None")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        plt.plot(
            self.res_ob.delta["mean_comp"],
            np.linspace(0, 1, self.n_delta + 2)[1:-1],
            color=colors[0],
            label=legend_labels[0],
            **kwargs,
        )
        plt.plot(
            self.res_ob.delta["mean_incomp"],
            np.linspace(0, 1, self.n_delta + 2)[1:-1],
            color=colors[1],
            label=legend_labels[1],
            **kwargs,
        )

        kwargs["linestyle"] = "-"
        kwargs["marker"] = "None"
        plt.plot(
            self.res_th.delta["mean_comp"],
            np.linspace(0, 1, self.n_delta + 2)[1:-1],
            color=colors[0],
            label=legend_labels[2],
            **kwargs,
        )
        plt.plot(
            self.res_th.delta["mean_incomp"],
            np.linspace(0, 1, self.n_delta + 2)[1:-1],
            color=colors[1],
            label=legend_labels[3],
            **kwargs,
        )

        if xlim is None:
            xlim = [
                np.min(self.res_ob.delta.mean_bin) - 100,
                np.max(self.res_ob.delta.mean_bin) + 100,
            ]

        plt.margins(x=0.5)
        _adjust_plt(xlim, None, xlabel, ylabel, label_fontsize, tick_fontsize)

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
        legend_labels=(
            "Compatible Observed",
            "Incompatible Observed",
            "Compatible Predicted",
            "Incompatible Predicted",
        ),
        legend_position="lower right",
        colors=("green", "red"),
        **kwargs,
    ):
        """

        Parameters
        ----------
        show
        ylim
        xlabel
        ylabel
        label_fontsize
        tick_fontsize
        legend_labels
        legend_position
        colors
        kwargs
        """
        kwargs.setdefault("linestyle", "None")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        plt.plot(
            self.res_ob.caf["bin"][self.res_ob.caf["Comp"] == "comp"],
            self.res_ob.caf["Error"][self.res_ob.caf["Comp"] == "comp"],
            color=colors[0],
            label=legend_labels[0],
            **kwargs,
        )
        plt.plot(
            self.res_ob.caf["bin"][self.res_ob.caf["Comp"] == "incomp"],
            self.res_ob.caf["Error"][self.res_ob.caf["Comp"] == "incomp"],
            color=colors[1],
            label=legend_labels[1],
            **kwargs,
        )

        kwargs["linestyle"] = "-"
        kwargs["marker"] = "None"
        plt.plot(
            self.res_th.caf["bin"][self.res_th.caf["Comp"] == "comp"],
            self.res_th.caf["Error"][self.res_th.caf["Comp"] == "comp"],
            color=colors[0],
            label=legend_labels[0],
            **kwargs,
        )
        plt.plot(
            self.res_th.caf["bin"][self.res_th.caf["Comp"] == "incomp"],
            self.res_th.caf["Error"][self.res_th.caf["Comp"] == "incomp"],
            color=colors[1],
            label=legend_labels[1],
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
        legend_labels=("Observed", "Predicted"),
        legend_position="lower right",
        **kwargs,
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
        legend_labels
        legend_position
        kwargs
        """

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)
        kwargs.setdefault("linestyle", "None")

        plt.plot(
            self.res_ob.delta["mean_bin"],
            self.res_ob.delta["mean_effect"],
            label=legend_labels[0],
            **kwargs,
        )

        kwargs["linestyle"] = "-"
        kwargs["marker"] = "None"
        plt.plot(
            self.res_th.delta["mean_bin"],
            self.res_th.delta["mean_effect"],
            label=legend_labels[1],
            **kwargs,
        )

        xlim = xlim or [
            np.min(self.res_ob.delta.mean_bin) - 100,
            np.max(self.res_ob.delta.mean_bin) + 100,
        ]
        ylim = ylim or [
            np.min(self.res_ob.delta.mean_effect) - 25,
            np.max(self.res_ob.delta.mean_effect) + 25,
        ]
        _adjust_plt(xlim, ylim, xlabel, ylabel, label_fontsize, tick_fontsize)

        if legend_position:
            plt.legend(loc=legend_position)

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
