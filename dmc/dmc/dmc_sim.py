"""
DMC model simulation detailed in  Ulrich, R., Schröter, H., Leuthold, H.,
    & Birngruber, T. (2015). Automatic and controlled stimulus processing
    in conflict tasks: Superimposed diffusion processes and delta functions.
    Cognitive Psychology, 78, 148-174.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import distplot


def simulation(amp=20, tau=30, aa_shape=2, mu=0.5, sigma=4, bnds=75,
               res_mean=300, res_sd=30, num_trls=100000, t_max=1000,
               var_dr=False, dr_lim=(0.1, 0.7), dr_shape=3,
               var_sp=False, sp_shape=3, rand_nums=None, plt_figs=True):
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
    num_trls: int (1000 to 100000 (default)), optional
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
    rand_nums: ndarray, optional
        pre-generated random array of size num_trls * t_max
    plt_figs: bool, optional
        plot figures

    Returns
    -------
    - Pandas dataframe with results summary
    - Figures 3 (default), 4 and 5 with changes in tau (see Table 1)

    Notes
    -----
    - Tested with Python 3.6
    - Code adapted from Appendix C. Basic Matlab Code.

    Examples
    --------
    >>> from dmc import dmc_sim
    >>> dmc_sim.simulation()               # Fig 3
    >>> dmc_sim.simulation(tau = 150)      # Fig 4
    >>> dmc_sim.simulation(tau = 90)       # Fig 5
    >>> dmc_sim.simulation(var_sp = True)  # Fig 6
    >>> dmc_sim.simulation(var_dr = True)  # Fig 7

    """

    # simulation
    if rand_nums is None:
        rand_nums = np.random.randn(num_trls, t_max)

    tim = np.arange(1, t_max + 1, 1)
    eq4 = amp * np.exp(-tim / tau) * (np.exp(1) * tim /
                                      (aa_shape - 1) / tau) ** (aa_shape - 1)

    # constant vs. variable drift rate
    if not var_dr:
        mu_c = eq4 * ((aa_shape - 1) / tim - 1 / tau) + mu
        mu_i = -eq4 * ((aa_shape - 1) / tim - 1 / tau) + mu
    else:
        drifts = rand_beta(dr_lim, dr_shape, num_trls)
        mu_c = eq4 * ((aa_shape-1)/tim-1/tau) + np.tile(drifts, (t_max, 1)).T

        drifts = rand_beta(dr_lim, dr_shape, num_trls)
        mu_i = -eq4 * ((aa_shape-1)/tim-1/tau) + np.tile(drifts, (t_max, 1)).T

    # random process
    x_c = mu_c + (sigma * rand_nums)
    np.random.shuffle(rand_nums)
    x_i = mu_i + (sigma * rand_nums)

    # variable starting point
    if var_sp:
        x_c[:, 0] += rand_beta((-bnds, bnds), sp_shape, num_trls)
        x_i[:, 0] += rand_beta((-bnds, bnds), sp_shape, num_trls)

    # cumulate activation
    x_c = np.cumsum(x_c, 1)
    x_i = np.cumsum(x_i, 1)

    # reaction time
    rt_c = np.argmax(np.abs(x_c) >= bnds, axis=1) + 1
    rt_i = np.argmax(np.abs(x_i) >= bnds, axis=1) + 1

    rt_c = np.vstack((rt_c + np.random.normal(res_mean, res_sd, num_trls),
                      x_c[np.arange(len(x_c)), rt_c - 1] < 0))
    rt_i = np.vstack((rt_i + np.random.normal(res_mean, res_sd, num_trls),
                      x_i[np.arange(len(x_c)), rt_i - 1] < 0))

    # calculate conditional accuracy function (CAF) values
    caf_c = calc_caf_values(rt_c)
    caf_i = calc_caf_values(rt_i)
    
    # calculate compatibility effect + delta values for correct trials
    time_delta, effect_delta = calc_delta_values(rt_c[0], rt_i[0])

    # figures
    if plt_figs:

        # upper left panel
        plt.subplot2grid((6, 4), (0, 0), rowspan=3, colspan=2)
        plt.plot(eq4, 'k')
        plt.plot(eq4 * -1, 'k--')
        plt.plot(x_c.mean(0), 'g')
        plt.plot(x_i.mean(0), 'r')
        plt.plot(np.cumsum(np.repeat(mu, t_max)), 'k')
        plt.xlim([0, t_max])
        plt.ylim([-bnds - 20, bnds + 20])
        plt.yticks((-bnds, bnds))

        # lower left panel
        plt.subplot2grid((6, 4), (3, 0), rowspan=3, colspan=2)
        for trl in range(5):
            idx = np.where(np.abs(x_c[trl, :] >= bnds))[0][0]
            plt.plot(x_c[trl][0:idx], 'g')
            idx = np.where(np.abs(x_i[trl, :] >= bnds))[0][0]
            plt.plot(x_i[trl][0:idx], 'r')
        plt.plot([0, t_max], [bnds, bnds], 'k--')
        plt.plot([0, t_max], [-bnds, -bnds], 'k--')
        plt.xlim([0, t_max])
        plt.ylim([-bnds - 20, bnds + 20])
        plt.yticks((-bnds, bnds))
        plt.xlabel("Time (ms)")
        plt.ylabel("X(t)")

        # upper right panel (left)
        plt.subplot2grid((6, 4), (0, 2), rowspan=2)
        distplot(np.random.choice(rt_c[0], t_max),
                 hist=False, kde_kws={"color": "g"})
        distplot(np.random.choice(rt_i[0], t_max),
                 hist=False, kde_kws={"color": "r"})
        plt.xlim([0, 1000])
        plt.ylim([0, 0.01])
        plt.ylabel("PDF")

        # upper right panel (right)
        plt.subplot2grid((6, 4), (0, 3), rowspan=2)
        distplot(np.random.choice(rt_c[0], t_max),
                 hist=False, kde_kws={"cumulative": True, "color": "g"})
        distplot(np.random.choice(rt_i[0], t_max),
                 hist=False, kde_kws={"cumulative": True, "color": "r"})
        plt.xlim([0, t_max])
        plt.ylim([0, 1.01])
        plt.ylabel("CDF")

        # middle left panel 
        plt.subplot2grid((6, 4), (2, 2), rowspan=2, colspan=2)
        plt.plot(caf_c, 'go-')
        plt.plot(caf_i, 'ro-')
        plt.ylim([0, 1.01])
        plt.xlabel("RT Bin")
        plt.xticks(np.arange(0, 5), ["1", "2", "3", "4", "5"])
        plt.ylabel("CAF")

        # bottom right panel
        plt.subplot2grid((6, 4), (4, 2), rowspan=2, colspan=2)
        plt.plot(time_delta, effect_delta, 'ko-', markersize=4)
        plt.xlim([0, t_max])
        plt.ylim([-50, 100])
        plt.xlabel("Time (ms)")
        plt.ylabel(r"$\Delta$")

        plt.subplots_adjust(hspace=1.5, wspace=0.35)
        plt.show(block=False)

    # create results table
    return results_summary(rt_c, rt_i, num_trls)


def rand_beta(lim=(0, 1), shape=3, num_trls=1):
    """Return random vector between limits weighted by beta function."""
    x = np.random.beta(shape, shape, num_trls)
    x = x * (lim[1] - lim[0]) + lim[0]
    
    return x
    
    
def calc_caf_values(dat, bounds=range(20, 100, 20)):
    """Calculate conditional accuracy functions"""
    
    bins = np.digitize(dat[0, :], np.percentile(dat[0], bounds))
    caf = []
    for b in np.unique(bins):
        idx = np.where(bins == b)
        n_obs = len(idx[0])
        caf.append(1 - (sum(dat[1][idx])/n_obs))
    
    return caf
    

def calc_delta_values(rt_c, rt_i, bounds=range(10, 100, 10)):
    """Calculate compatibility effect + delta values for correct trials."""
    bins_c = np.percentile(rt_c, bounds)
    bins_i = np.percentile(rt_i, bounds)

    return (bins_c + bins_i) / 2, bins_i - bins_c


def results_summary(rt_c, rt_i, n_trl):
    """Create results summary table."""
    res = [[round(np.mean(rt_c[0][rt_c[1] == 0])),
            round(np.std(rt_c[0][rt_c[1] == 0])),
            round(np.sum(rt_c[1] / n_trl) * 100, 1),
            round(np.mean(rt_c[0][rt_c[1] == 1])),
            round(np.std(rt_c[0][rt_c[1] == 1]))],
           [round(np.mean(rt_i[0][rt_i[1] == 0])),
            round(np.std(rt_i[0][rt_i[1] == 0])),
            round(np.sum(rt_i[1] / n_trl) * 100, 1),
            round(np.mean(rt_i[0][rt_i[1] == 1])),
            round(np.std(rt_i[0][rt_i[1] == 1]))]]

    res = pd.DataFrame(res, index=['comp', 'incomp'],
                       columns=['rtCorr', 'sdCorr', 'perErr',
                                'rtErr', 'sdRtErr'])

    return res


if __name__ == "__main__":
    simulation()
