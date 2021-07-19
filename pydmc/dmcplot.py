import inspect
import matplotlib.pyplot as plt
import numpy as np
from fastkde import fastKDE


class DmcPlot:
    def __init__(self, dat, fig_type="summary1"):
        self.dat = dat
        self.fig_type = fig_type

    def plot(self, **kwargs):
        """Plot summary"""
        if hasattr(self.dat, "prms"):
            if self.fig_type == "summary1" and not self.dat.full_data:
                self.fig_type = "summary2"
            if self.fig_type == "summary1":
                self._plot_summary1_res_th(**kwargs)
            elif self.fig_type == "summary2":
                self._plot_summary2_res_th(**kwargs)
            elif self.fig_type == "summary3":
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

        plt.subplots_adjust(
            hspace=kwargs.get("hspace", 0.5), wspace=kwargs.get("wspace", 0.5)
        )
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

        plt.subplots_adjust(
            hspace=kwargs.get("hspace", 0.5), wspace=kwargs.get("wspace", 0.5)
        )
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

        plt.subplots_adjust(
            hspace=kwargs.get("hspace", 0.5), wspace=kwargs.get("wspace", 0.5)
        )
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

        plt.subplots_adjust(
            hspace=kwargs.get("hspace", 0.5), wspace=kwargs.get("wspace", 0.5)
        )
        plt.show(block=False)

    def plot_activation(
        self,
        show=True,
        xlim=None,
        ylim=None,
        xlabel="Time (ms)",
        ylabel="E[X(t)]",
        cond_labels=("Compatible", "Incompatible"),
        legend_position="best",
        colors=("green", "red"),
        **kwargs
    ):
        """Plot activation"""

        if not self.dat.xt:
            print("Plotting activation function requires full_data=True")
            return

        plt.plot(self.dat.eq4, "k-")
        plt.plot(self.dat.eq4 * -1, "k--")

        l_kws = _filter_dict(kwargs, plt.Line2D)

        plt.plot(self.dat.xt[0], color=colors[0], label=cond_labels[0], **l_kws)
        plt.plot(self.dat.xt[1], color=colors[1], label=cond_labels[1], **l_kws)
        plt.plot(
            np.cumsum(np.repeat(self.dat.prms.drc, self.dat.prms.t_max)),
            color="black",
            **l_kws,
        )

        kwargs["xlabel"] = xlabel
        kwargs["ylabel"] = ylabel
        kwargs["xlim"] = xlim or [0, self.dat.prms.t_max]
        kwargs["ylim"] = ylim or [-self.dat.prms.bnds - 20, self.dat.prms.bnds + 20]
        self._plot_bounds()
        _adjust_plt(**kwargs)

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
        cond_labels
        legend_position
        colors
        kwargs
        """

        if not self.dat.xt:
            print("Plotting individual trials function requires full_data=True")
            return

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for trl in range(self.dat.n_trls_data):
            if trl == 0:
                labels = cond_labels
            else:
                labels = [None, None]
            idx = np.where(
                np.abs(self.dat.dat_trials[0][trl, :]) >= self.dat.prms.bnds
            )[0][0]
            plt.plot(
                self.dat.dat_trials[0][trl][0:idx],
                color=colors[0],
                label=labels[0],
                **l_kws,
            )
            idx = np.where(
                np.abs(self.dat.dat_trials[1][trl, :]) >= self.dat.prms.bnds
            )[0][0]
            plt.plot(
                self.dat.dat_trials[1][trl][0:idx],
                color=colors[1],
                label=labels[1],
                **l_kws,
            )

        xlim = xlim or [0, self.dat.prms.t_max]
        ylim = ylim or [-self.dat.prms.bnds - 20, self.dat.prms.bnds + 20]
        self._plot_bounds()
        _adjust_plt(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, **kwargs)

        if legend_position:
            plt.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def _plot_bounds(self):
        plt.axhline(y=self.dat.prms.bnds, color="black", linestyle="--")
        plt.axhline(y=-self.dat.prms.bnds, color="black", linestyle="--")

    def plot_pdf(
        self,
        show=True,
        xlim=None,
        ylim=None,
        xlabel="Time (ms)",
        ylabel="PDF",
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
        cond_labels
        legend_position
        colors
        kwargs
        """

        comp_pdf, axes1 = fastKDE.pdf(self.dat.dat[0][0])
        incomp_pdf, axes2 = fastKDE.pdf(self.dat.dat[1][0])

        l_kws = _filter_dict(kwargs, plt.Line2D)
        plt.plot(axes1, comp_pdf, color=colors[0], label=cond_labels[0], **l_kws)
        plt.plot(axes2, incomp_pdf, color=colors[1], label=cond_labels[1], **l_kws)

        xlim = xlim or [0, self.dat.prms.t_max]
        ylim = ylim or [0, 0.01]
        _adjust_plt(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, **kwargs)

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
        cond_labels
        legend_position
        colors
        kwargs
        """

        l_kws = _filter_dict(kwargs, plt.Line2D)
        if hasattr(self.dat, "prms"):
            for comp in (0, 1):
                pdf, axes = fastKDE.pdf(self.dat.dat[comp][0])
                plt.plot(
                    axes,
                    np.cumsum(pdf) * np.diff(axes)[0:1],
                    color=colors[comp],
                    label=cond_labels[comp],
                    **l_kws,
                )

            xlim = xlim or [0, self.dat.prms.t_max]
            ylim = ylim or [0, 1.05]

        if not hasattr(self.dat, "prms"):

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

            xlim = xlim or [
                np.min(self.dat.mean_bin) - 100,
                np.max(self.dat.delta.mean_bin) + 100,
            ]
            ylim = ylim or [0, 1.05]

        _adjust_plt(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, **kwargs)

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
        cond_labels
        legend_position
        colors
        kwargs
        """

        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, comp in enumerate(("comp", "incomp")):
            plt.plot(
                self.dat.caf["bin"][self.dat.caf["Comp"] == comp],
                self.dat.caf["Error"][self.dat.caf["Comp"] == comp],
                color=colors[idx],
                label=cond_labels[idx],
                **l_kws,
            )

        plt.xticks(
            range(1, self.dat.n_caf + 1), [str(x) for x in range(1, self.dat.n_caf + 1)]
        )
        _adjust_plt(ylim=ylim, xlabel=xlabel, ylabel=ylabel, **kwargs)

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
        kwargs
        """

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        datx, daty = self.dat.delta["mean_bin"], self.dat.delta["mean_effect"]

        xlim = xlim or [np.min(datx) - 100, np.max(datx) + 100]
        ylim = ylim or [np.min(daty) - 25, np.max(daty) + 25]

        l_kws = _filter_dict(kwargs, plt.Line2D)
        plt.plot(datx, daty, **l_kws)
        _adjust_plt(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, **kwargs)

        if show:
            plt.show(block=False)

    def plot_rt_correct(
        self,
        show=True,
        ylim=None,
        xlabel=None,
        ylabel="RT Correct [ms]",
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
        cond_labels
        kwargs
        """

        _plot_beh(
            self.dat.summary["rt_cor"],
            cond_labels,
            ylim,
            xlabel,
            ylabel,
            False,
            **kwargs,
        )

        if show:
            plt.show(block=False)

    def plot_er(
        self,
        show=True,
        ylim=None,
        xlabel=None,
        ylabel="Error Rate [%]",
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

        _plot_beh(
            self.dat.summary["per_err"],
            cond_labels,
            ylim,
            xlabel,
            ylabel,
            True,
            **kwargs,
        )

        if show:
            plt.show(block=False)

    def plot_rt_error(
        self,
        show=True,
        ylim=None,
        xlabel=None,
        ylabel="RT Error [ms]",
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
        cond_labels
        kwargs
        """

        _plot_beh(
            self.dat.summary["rt_err"],
            cond_labels,
            ylim,
            xlabel,
            ylabel,
            False,
            **kwargs,
        )

        if show:
            plt.show(block=False)


def _plot_beh(dat, cond_labels, ylim, xlabel, ylabel, zeroed, **kwargs):
    """Internal function to plot rt/er for comp vs. comp"""

    kwargs.setdefault("color", "black")
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", 4)

    l_kws = _filter_dict(kwargs, plt.Line2D)
    plt.plot(cond_labels, dat, **l_kws)
    plt.margins(x=0.5)

    if zeroed:
        ylim = ylim or [0, np.max(dat) + 5]
    else:
        ylim = ylim or [np.min(dat) - 100, np.max(dat) + 100]

    _adjust_plt(ylim=ylim, xlabel=xlabel, ylabel=ylabel, **kwargs)


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
