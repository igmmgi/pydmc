import inspect
import matplotlib.pyplot as plt
import numpy as np
from fastkde import fastKDE


class DmcPlot:
    def __init__(self, res):
        self.res = res

    def plot(self, **kwargs):
        """Plot summary."""
        kwargs.setdefault("fig_type", "summary1")
        kwargs.setdefault("hspace", 0.5)
        kwargs.setdefault("wspace", 0.5)
        if hasattr(self.res, "prms"):
            if kwargs["fig_type"] == "summary1" and not self.res.full_data:
                kwargs["fig_type"] = "summary2"
            if kwargs["fig_type"] == "summary1":
                self._plot_summary1_res_th(**kwargs)
            elif kwargs["fig_type"] == "summary2":
                self._plot_summary2_res_th(**kwargs)
            elif kwargs["fig_type"] == "summary3":
                self._plot_summary3_res_th(**kwargs)
        else:
            self._plot_summary1_res_ob(**kwargs)

    def _plot_summary1_res_th(self, **kwargs):

        # upper left panel (activation)
        plt.subplot2grid((6, 4), (0, 0), rowspan=3, colspan=2)
        self.plot_activation(show=False, **kwargs)

        # lower left panel (trials)
        plt.subplot2grid((6, 4), (3, 0), rowspan=3, colspan=2)
        self.plot_trials(show=False, **kwargs)

        # upper right (left) panel (PDF)
        plt.subplot2grid((6, 4), (0, 2), rowspan=2)
        self.plot_pdf(show=False, **kwargs)

        # upper right (right) panel (CDF)
        plt.subplot2grid((6, 4), (0, 3), rowspan=2)
        self.plot_cdf(show=False, **kwargs)

        # middle right panel (CAF)
        plt.subplot2grid((6, 4), (2, 2), rowspan=2, colspan=2)
        self.plot_caf(show=False, **kwargs)

        # bottom right panel (delta)
        plt.subplot2grid((6, 4), (4, 2), rowspan=2, colspan=2)
        self.plot_delta(show=False, **kwargs)

        plt.subplots_adjust(hspace=kwargs["hspace"], wspace=kwargs["wspace"])
        plt.show(block=False)

    def _plot_summary2_res_th(self, **kwargs):

        # upper right (left) panel (PDF)
        plt.subplot2grid((3, 2), (0, 0))
        self.plot_pdf(show=False, **kwargs)

        # upper right (eight) panel (CDF)
        plt.subplot2grid((3, 2), (0, 1))
        self.plot_cdf(show=False, **kwargs)

        # middle left panel
        plt.subplot2grid((3, 2), (1, 0), colspan=2)
        self.plot_caf(show=False, **kwargs)

        # bottom right panel
        plt.subplot2grid((3, 2), (2, 0), colspan=2)
        self.plot_delta(show=False, **kwargs)

        plt.subplots_adjust(hspace=kwargs["hspace"], wspace=kwargs["wspace"])
        plt.show(block=False)

    def _plot_summary3_res_th(self, **kwargs):

        # upper right (left) panel (PDF)
        plt.subplot2grid((3, 1), (0, 0))
        self.plot_rt_correct(show=False, **kwargs)

        # upper right (eight) panel (CDF)
        plt.subplot2grid((3, 1), (1, 0))
        self.plot_er(show=False, **kwargs)

        # middle left panel
        plt.subplot2grid((3, 1), (2, 0))
        self.plot_rt_error(show=False, **kwargs)

        plt.subplots_adjust(hspace=kwargs["hspace"], wspace=kwargs["wspace"])
        plt.show(block=False)

    def _plot_summary1_res_ob(self, **kwargs):
        """Plot summaty observed data."""

        # upper left panel (rt correct)
        plt.subplot2grid((3, 2), (0, 0))
        self.plot_rt_correct(show=False, **kwargs)

        # middle left pannel
        plt.subplot2grid((3, 2), (1, 0))
        self.plot_er(show=False, **kwargs)

        # bottom left pannel
        plt.subplot2grid((3, 2), (2, 0))
        self.plot_rt_error(show=False, **kwargs)

        # upper right panel (cdf)
        plt.subplot2grid((3, 2), (0, 1))
        self.plot_cdf(show=False, **kwargs)

        # middle right (left) panel (PDF)
        plt.subplot2grid((3, 2), (1, 1))
        self.plot_caf(show=False, **kwargs)

        # lower right (right) panel (CDF)
        plt.subplot2grid((3, 2), (2, 1))
        self.plot_delta(show=False, **kwargs)

        plt.subplots_adjust(hspace=kwargs["hspace"], wspace=kwargs["wspace"])
        plt.show(block=False)

    def plot_activation(
        self,
        show=True,
        cond_labels=("Compatible", "Incompatible"),
        legend_position="best",
        colors=("green", "red"),
        **kwargs,
    ):
        """Plot activation."""

        if not self.res.xt:
            print("Plotting activation function requires full_data=True")
            return

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
        self._plot_bounds()

        kwargs.setdefault("xlim", [0, self.res.prms.t_max])
        kwargs.setdefault("ylim", [-self.res.prms.bnds - 20, self.res.prms.bnds + 20])
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", "E[X(t)]")

        _adjust_plt(**kwargs)

        if legend_position is not None:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def plot_trials(
        self,
        show=True,
        cond_labels=("Compatible", "Incompatible"),
        legend_position="upper right",
        colors=("green", "red"),
        **kwargs,
    ):
        """Plot individual trials."""

        if not self.res.xt:
            print("Plotting individual trials function requires full_data=True")
            return

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for trl in range(self.res.n_trls_data):
            if trl == 0:
                labels = cond_labels
            else:
                labels = [None, None]
            for comp in (0, 1):
                idx = np.where(
                    np.abs(self.res.dat_trials[comp][trl, :]) >= self.res.prms.bnds
                )[0][0]
                plt.plot(
                    self.res.dat_trials[comp][trl][0:idx],
                    color=colors[comp],
                    label=labels[comp],
                    **l_kws,
                )
        self._plot_bounds()

        kwargs.setdefault("xlim", [0, self.dat.prms.t_max])
        kwargs.setdefault("ylim", [-self.res.prms.bnds - 20, self.res.prms.bnds + 20])
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", "X(t)")

        _adjust_plt(**kwargs)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def _plot_bounds(self):
        plt.axhline(y=self.res.prms.bnds, color="black", linestyle="--")
        plt.axhline(y=-self.res.prms.bnds, color="black", linestyle="--")

    def plot_pdf(
        self,
        show=True,
        cond_labels=("Compatible", "Incompatible"),
        legend_position="upper right",
        colors=("green", "red"),
        **kwargs,
    ):
        """Plot PDF."""

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for comp in (0, 1):
            pdf, axes = fastKDE.pdf(self.res.dat[comp][0])
            plt.plot(axes, pdf, color=colors[comp], label=cond_labels[comp], **l_kws)

        kwargs.setdefault("xlim", [0, self.res.prms.t_max])
        kwargs.setdefault("ylim", [0, 0.01])
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", "PDF")

        _adjust_plt(**kwargs)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def plot_cdf(
        self,
        show=True,
        cond_labels=("Compatible", "Incompatible"),
        legend_position="lower right",
        colors=("green", "red"),
        **kwargs,
    ):
        """Plot CDF."""

        l_kws = _filter_dict(kwargs, plt.Line2D)
        if hasattr(self.res, "prms"):
            for comp in (0, 1):
                pdf, axes = fastKDE.pdf(self.res.dat[comp][0])
                plt.plot(
                    axes,
                    np.cumsum(pdf) * np.diff(axes)[0:1],
                    color=colors[comp],
                    label=cond_labels[comp],
                    **l_kws,
                )

            kwargs.setdefault("xlim", [0, self.res.prms.t_max])

        if not hasattr(self.res, "prms"):

            kwargs.setdefault("marker", "o")
            kwargs.setdefault("markersize", 4)

            for idx, comp in enumerate(("mean_comp", "mean_incomp")):
                plt.plot(
                    self.delta[comp],
                    np.linspace(0, 1, self.n_delta + 2)[1:-1],
                    color=colors[idx],
                    label=cond_labels[idx],
                    **l_kws,
                )

            kwargs.setdefault(
                "xlim",
                [
                    np.min(self.res.mean_bin) - 100,
                    np.max(self.res.delta.mean_bin) + 100,
                ],
            )

        kwargs.setdefault("ylim", [0, 1.05])
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", "CDF")

        _adjust_plt(**kwargs)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def plot_caf(
        self,
        show=True,
        cond_labels=("Compatible", "Incompatible"),
        legend_position="lower right",
        colors=("green", "red"),
        **kwargs,
    ):
        """Plot CAF."""

        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, comp in enumerate(("comp", "incomp")):
            plt.plot(
                self.res.caf["bin"][self.res.caf["Comp"] == comp],
                self.res.caf["Error"][self.res.caf["Comp"] == comp],
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

    def plot_delta(self, show=True, **kwargs):
        """Plot reaction-time delta plots."""

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        datx, daty = self.res.delta["mean_bin"], self.res.delta["mean_effect"]

        l_kws = _filter_dict(kwargs, plt.Line2D)
        plt.plot(datx, daty, **l_kws)

        kwargs.setdefault("xlim", [np.min(datx) - 100, np.max(datx) + 100])
        kwargs.setdefault("ylim", [np.min(daty) - 25, np.max(daty) + 25])
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", r"$\Delta$")

        _adjust_plt(**kwargs)

        if show:
            plt.show(block=False)

    def plot_rt_correct(
        self, show=True, cond_labels=("Compatible", "Incompatible"), **kwargs
    ):
        """Plot correct RT's."""

        kwargs.setdefault("ylabel", "RT Correct [ms]")

        _plot_beh(
            self.res.summary["rt_cor"],
            cond_labels,
            False,
            **kwargs,
        )

        if show:
            plt.show(block=False)

    def plot_er(self, show=True, cond_labels=("Compatible", "Incompatible"), **kwargs):
        """Plot error rate"""

        kwargs.setdefault("ylabel", "Error Rate [%]")

        _plot_beh(
            self.res.summary["per_err"],
            cond_labels,
            True,
            **kwargs,
        )

        if show:
            plt.show(block=False)

    def plot_rt_error(
        self, show=True, cond_labels=("Compatible", "Incompatible"), **kwargs
    ):
        """Plot error RT's."""

        kwargs.setdefault("ylabel", "RT Error [ms]")

        _plot_beh(
            self.res.summary["rt_err"],
            cond_labels,
            False,
            **kwargs,
        )

        if show:
            plt.show(block=False)


class DmcPlotFit:
    def __init__(self, res_th, res_ob):
        self.res_th = res_th
        self.res_ob = res_ob

    def plot(self, **kwargs):
        """Plot."""
        kwargs.setdefault("fig_type", "summary1")
        kwargs.setdefault("hspace", 0.5)
        kwargs.setdefault("wspace", 0.5)

        # upper left panel (rt correct)
        plt.subplot2grid((3, 2), (0, 0))
        self.plot_rt_correct(show=False, **kwargs)

        # middle left panel
        plt.subplot2grid((3, 2), (1, 0))
        self.plot_er(show=False, **kwargs)

        # bottom left panel
        plt.subplot2grid((3, 2), (2, 0))
        self.plot_rt_error(show=False, **kwargs)

        # upper right panel (CDF)
        plt.subplot2grid((3, 2), (0, 1))
        self.plot_cdf(show=False, **kwargs)

        # middle right panel (CAF)
        plt.subplot2grid((3, 2), (1, 1))
        self.plot_caf(show=False, **kwargs)

        # bottom right panel (delta)
        plt.subplot2grid((3, 2), (2, 1))
        self.plot_delta(show=False, **kwargs)

        plt.subplots_adjust(hspace=kwargs["hspace"], wspace=kwargs["wspace"])
        plt.show(block=False)

    def plot_rt_correct(
        self,
        show=True,
        cond_labels=("Compatible", "Incompatible"),
        legend_labels=("Observed", "Predicted"),
        legend_position="upper left",
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

    def plot_er(
        self,
        show=True,
        cond_labels=("Compatible", "Incompatible"),
        legend_labels=("Observed", "Predicted"),
        legend_position="upper left",
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

    def plot_rt_error(
        self,
        show=True,
        cond_labels=("Compatible", "Incompatible"),
        legend_labels=("Observed", "Predicted"),
        legend_position="upper left",
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

    def plot_cdf(
        self,
        show=True,
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

    def plot_caf(
        self,
        show=True,
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
        """Plot CAF."""

        kwargs.setdefault("linestyle", "None")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)
        kwargs.setdefault("xlabel", "RT Bin")
        kwargs.setdefault("ylabel", "CAF")

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, comp in enumerate(("comp", "incomp")):
            plt.plot(
                self.res_ob.caf["bin"][self.res_ob.caf["Comp"] == comp],
                self.res_ob.caf["Error"][self.res_ob.caf["Comp"] == comp],
                color=colors[idx],
                label=legend_labels[idx],
                **l_kws,
            )

        kwargs["linestyle"] = "-"
        kwargs["marker"] = "None"
        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, comp in enumerate(("comp", "incomp")):
            plt.plot(
                self.res_th.caf["bin"][self.res_th.caf["Comp"] == comp],
                self.res_th.caf["Error"][self.res_th.caf["Comp"] == comp],
                color=colors[idx],
                label=legend_labels[idx+2],
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

    def plot_delta(
        self,
        show=True,
        legend_labels=("Observed", "Predicted"),
        legend_position="lower right",
        **kwargs,
    ):
        """Plot reaction-time delta plots."""
        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)
        kwargs.setdefault("linestyle", "None")
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("xlabel", r"$\Delta$")

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


def _plot_beh(dat, cond_labels, zeroed, **kwargs):
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
    """Internal function to filter dict **kwargs"""
    f = {}
    allowed = inspect.signature(allowed).parameters
    for k, v in given.items():
        if k in allowed.keys():
            f[k] = v
    return f


def _adjust_plt(**kwargs):
    """Internal function to adjust some plot properties."""
    plt.xlim(kwargs.get("xlim", None))
    plt.ylim(kwargs.get("ylim", None))
    plt.xlabel(kwargs.get("xlabel", ""), fontsize=kwargs.get("label_fontsize", 12))
    plt.ylabel(kwargs.get("ylabel", ""), fontsize=kwargs.get("label_fontsize", 12))
    plt.xticks(fontsize=kwargs.get("tick_fontsize", 10))
    plt.yticks(fontsize=kwargs.get("tick_fontsize", 10))
