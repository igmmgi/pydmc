"""
DMC model simulation detailed in  Ulrich, R., Schröter, H., Leuthold, H., 
& Birngruber, T. (2015). Automatic and controlled stimulus processing
in conflict tasks: Superimposed diffusion processes and delta functions.
Cognitive Psychology, 78, 148-174.
"""
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange
import pandas as pd
from seaborn import distplot


class DMC:
    def __init__(
        self,
        amp=20,
        tau=30,
        aa_shape=2,
        mu=0.5,
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
        step_caf=range(20, 100, 20),
        step_delta=range(10, 100, 10),
        full_data=True,
        run_simulation=True,
        plt_figs=True,
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
        mu: int/float, optional
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
        step_caf: range
            caf bins
        step_delta: range
            delta reaction time bins

        Returns
        -------
        - Pandas dataframe with results summary
        - Figures 3 (default), 4 and 5 with changes in tau (see Table 1)

        Notes
        -----
        - Tested with Python 3.7
        - Code adapted from Appendix C. Basic Matlab Code.

        Examples
        --------
        >>> from dmc.dmc import DMC 
        >>> dat = DMC()                # Fig 3
        >>> dat = DMC(full_data=False) # Fig 3 (part)
        >>> dat = DMC(tau = 150)       # Fig 4
        >>> dat = DMC(tau = 90)        # Fig 5
        >>> dat = DMC(var_sp = True)   # Fig 6
        >>> dat = DMC(var_dr = True)   # Fig 7
        """

        self.amp = amp
        self.tau = tau
        self.aa_shape = aa_shape
        self.mu = mu
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
        self.step_caf = step_caf
        self.step_delta = step_delta
        self.full_data = full_data

        self.tim = np.arange(1, self.t_max + 1, 1)
        self.eq4 = (
            self.amp
            * np.exp(-self.tim / self.tau)
            * (np.exp(1) * self.tim / (self.aa_shape - 1) / self.tau)
            ** (self.aa_shape - 1)
        )
        self.dat = [None, None]
        self.xt = [None, None]

        if run_simulation:
            self.run_simulation()

        if plt_figs:
            self.plot()

    def run_simulation(self):
        """ Run simulation. """

        if self.full_data:
            self._run_simulation_numpy()
        else:
            self._run_simulation_numba()

    def _run_simulation_numpy(self):
        """ Run simulatin using numpy. """

        rand_nums = None
        for idx, comp in enumerate([1, -1]):
            if rand_nums is None:
                rand_nums = np.random.randn(self.n_trls, self.t_max)
            else:
                np.random.shuffle(rand_nums)

            dr = self._dr()
            sp = self._sp()
            mu = (
                comp * self.eq4 * ((self.aa_shape - 1) / self.tim - 1 / self.tau)
                + np.tile(dr, (self.t_max, 1)).T
            )

            # random process
            xt = mu + (self.sigma * rand_nums)

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

            self.xt[idx] = xt

        self._calc_caf_values()
        self._calc_delta_values()
        self._results_summary()

    def _run_simulation_numba(self):
        """ Run simulation using numba. """

        for idx, comp in enumerate([1, -1]):

            dr = self._dr()
            sp = self._sp()
            mu = comp * self.eq4 * ((self.aa_shape - 1) / self.tim - 1 / self.tau)

            self.dat[idx] = _run_simulation_numba(
                mu,
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
        """ Create results summary table. """

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
        """ Calculate conditional accuracy functions. """

        bc = np.digitize(self.dat[0][0], np.percentile(self.dat[0][0], self.step_caf))
        bi = np.digitize(self.dat[1][0], np.percentile(self.dat[1][0], self.step_caf))
        self.caf = np.zeros((2, len(np.unique(bc))))
        for bidx, bedge in enumerate(np.unique(bc)):

            idx = np.where(bc == bedge)
            self.caf[0][bidx] = 1 - (sum(self.dat[0][1][idx]) / len(idx[0]))

            idx = np.where(bi == bedge)
            self.caf[1][bidx] = 1 - (sum(self.dat[1][1][idx]) / len(idx[0]))

    def _calc_delta_values(self):
        """ Calculate compatibility effect + delta values for correct trials. """
        bc = np.percentile(self.dat[0][0], self.step_delta)
        bi = np.percentile(self.dat[1][0], self.step_delta)

        self.time_delta = (bc + bi) / 2
        self.effect_delta = bi - bc

    @staticmethod
    def rand_beta(lim=(0, 1), shape=3, n_trls=1):
        """ Return random vector between limits weighted by beta function. """
        x = np.random.beta(shape, shape, n_trls)
        x = x * (lim[1] - lim[0]) + lim[0]

        return x

    def plot(self):
        """ Plot """
        if self.full_data:
            self._plot_full()
        else:
            self._plot()

    def _plot_full(self):

        # upper left panel (activation)
        plt.subplot2grid((6, 4), (0, 0), rowspan=3, colspan=2)
        self.plot_activation()

        # lower left panel (trials)
        plt.subplot2grid((6, 4), (3, 0), rowspan=3, colspan=2)
        self.plot_trials()

        # upper right (left) panel (PDF)
        plt.subplot2grid((6, 4), (0, 2), rowspan=2)
        self.plot_pdf()

        # upper right (right) panel (CDF)
        plt.subplot2grid((6, 4), (0, 3), rowspan=2)
        self.plot_cdf()

        # middle riight panel (CAF)
        plt.subplot2grid((6, 4), (2, 2), rowspan=2, colspan=2)
        self.plot_caf()

        # bottom right panel (delta)
        plt.subplot2grid((6, 4), (4, 2), rowspan=2, colspan=2)
        self.plot_delta()

        plt.subplots_adjust(hspace=1.5, wspace=0.35)
        plt.show(block=False)

    def _plot(self):

        # upper right (left) panel (PDF)
        plt.subplot2grid((3, 2), (0, 0), rowspan=1)
        self.plot_pdf()

        # upper right (eight) panel (CDF)
        plt.subplot2grid((3, 2), (0, 1), rowspan=1)
        self.plot_cdf()

        # middle left panel
        plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=2)
        self.plot_caf()

        # bottom right panel
        plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=2)
        self.plot_delta()

        plt.subplots_adjust(hspace=1.5, wspace=0.35)
        plt.show(block=False)

    def plot_activation(self):
        """ Plot activation. """

        plt.plot(self.eq4, "k")
        plt.plot(self.eq4 * -1, "k--")
        plt.plot(self.xt[0].mean(0), "g")
        plt.plot(self.xt[1].mean(0), "r")
        plt.plot(np.cumsum(np.repeat(self.mu, self.t_max)), "k")
        plt.xlim([0, self.t_max])
        plt.ylim([-self.bnds - 20, self.bnds + 20])
        plt.yticks((-self.bnds, self.bnds))

    def plot_trials(self):
        """ Plot individual trials. """

        for trl in range(5):
            idx = np.where(np.abs(self.xt[0][trl, :] >= self.bnds))[0][0]
            plt.plot(self.xt[0][trl][0:idx], "g")
            idx = np.where(np.abs(self.xt[1][trl, :] >= self.bnds))[0][0]
            plt.plot(self.xt[1][trl][0:idx], "r")
        plt.plot([0, self.t_max], [self.bnds, self.bnds], "k--")
        plt.plot([0, self.t_max], [-self.bnds, -self.bnds], "k--")
        plt.xlim([0, self.t_max])
        plt.ylim([-self.bnds - 20, self.bnds + 20])
        plt.yticks((-self.bnds, self.bnds))
        plt.xlabel("Time (ms)")
        plt.ylabel("X(t)")

    def plot_pdf(self):
        """ Plot PDF. """
        distplot(
            np.random.choice(self.dat[0][0], self.t_max),
            hist=False,
            kde_kws={"color": "g"},
        )
        distplot(
            np.random.choice(self.dat[1][0], self.t_max),
            hist=False,
            kde_kws={"color": "r"},
        )
        plt.xlim([0, self.t_max])
        plt.ylim([0, 0.01])
        plt.ylabel("PDF")

    def plot_cdf(self):
        """ Plot CDF. """
        distplot(
            np.random.choice(self.dat[0][0], self.t_max),
            hist=False,
            kde_kws={"cumulative": True, "color": "g"},
        )
        distplot(
            np.random.choice(self.dat[1][0], self.t_max),
            hist=False,
            kde_kws={"cumulative": True, "color": "r"},
        )
        plt.xlim([0, self.t_max])
        plt.ylim([0, 1.01])
        plt.ylabel("CDF")

    def plot_caf(self):
        """ Plot CAF. """
        plt.plot(self.caf[0], "go-")
        plt.plot(self.caf[1], "ro-")
        plt.ylim([0, 1.01])
        plt.xlabel("RT Bin")
        plt.xticks(np.arange(0, 5), ["1", "2", "3", "4", "5"])
        plt.ylabel("CAF")

    def plot_delta(self):
        """ Plot reaction-time delta plots. """

        plt.plot(self.time_delta, self.effect_delta, "ko-", markersize=4)
        plt.xlim([0, self.t_max])
        plt.ylim([-50, 100])
        plt.xlabel("Time (ms)")
        plt.ylabel(r"$\Delta$")

    def _dr(self):
        if self.var_dr:
            return self.rand_beta(self.dr_lim, self.dr_shape, self.n_trls)
        else:
            return np.ones(self.n_trls) * self.mu

    def _sp(self):
        if self.var_sp:
            return self.rand_beta((-self.bnds, self.bnds), self.sp_shape, self.n_trls)
        else:
            return np.zeros(self.n_trls)


@jit(nopython=True, parallel=True)
def _run_simulation_numba(mu, sp, dr, t_max, sigma, res_mean, res_sd, bnds, n_trls):

    dat = np.vstack((np.ones(n_trls) * t_max, np.ones(n_trls)))

    for trl in prange(n_trls):
        trl_xt = sp[trl]
        for t in range(0, t_max):
            trl_xt += mu[t] + dr[trl] + (sigma * np.random.randn())
            if np.abs(trl_xt) >= bnds:
                dat[0][trl] = t + np.random.normal(res_mean, res_sd)
                if trl_xt > 0.0:
                    dat[1][trl] = False
                break

    return dat


if __name__ == "__main__":
    dmc = DMC()
