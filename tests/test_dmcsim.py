""" Basic unittests for DMC """
import unittest
from unittest import TestCase

from pydmc.dmcsim import DmcSim


class DMCTestCaseDmcSim(unittest.TestCase):
    """Tests for DmcSim()"""

    rt_tolerance = 5  # ms
    sd_tolerance = 5  # ms
    err_tolerance = 0.5  # %

    def test_dmcsim1(self):
        """
        Simulation 1 (Figure 3)
        amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = DmcSim(tau=30)

        self.assertLess(abs(440 - dat.summary["rt_cor"][0]), self.rt_tolerance)
        self.assertLess(abs(106 - dat.summary["sd_cor"][0]), self.sd_tolerance)
        self.assertLess(abs(0.7 - dat.summary["per_err"][0]), self.err_tolerance)
        self.assertLess(abs(458 - dat.summary["rt_cor"][1]), self.rt_tolerance)
        self.assertLess(abs(95 - dat.summary["sd_cor"][1]), self.sd_tolerance)
        self.assertLess(abs(1.4 - dat.summary["per_err"][1]), self.err_tolerance)

    def test_dmcsim2(self):
        """
        Simulation 1 (Figure 3)
        amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = DmcSim(tau=30, full_data=False)

        self.assertLess(abs(440 - dat.summary["rt_cor"][0]), self.rt_tolerance)
        self.assertLess(abs(106 - dat.summary["sd_cor"][0]), self.sd_tolerance)
        self.assertLess(abs(0.7 - dat.summary["per_err"][0]), self.err_tolerance)
        self.assertLess(abs(458 - dat.summary["rt_cor"][1]), self.rt_tolerance)
        self.assertLess(abs(95 - dat.summary["sd_cor"][1]), self.sd_tolerance)
        self.assertLess(abs(1.4 - dat.summary["per_err"][1]), self.err_tolerance)

    def test_dmcsim3(self):
        """
        Simulation 2 (Figure 4)
        amp = 20, tau = 150, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = DmcSim(tau=150)

        self.assertLess(abs(422 - dat.summary["rt_cor"][0]), self.rt_tolerance)
        self.assertLess(abs(90 - dat.summary["sd_cor"][0]), self.sd_tolerance)
        self.assertLess(abs(0.3 - dat.summary["per_err"][0]), self.err_tolerance)
        self.assertLess(abs(483 - dat.summary["rt_cor"][1]), self.rt_tolerance)
        self.assertLess(abs(103 - dat.summary["sd_cor"][1]), self.sd_tolerance)
        self.assertLess(abs(2.2 - dat.summary["per_err"][1]), self.err_tolerance)

    def test_dmcsim4(self):
        """
        Simulation 2 (Figure 4)
        amp = 20, tau = 150, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = DmcSim(tau=150, full_data=False)

        self.assertLess(abs(422 - dat.summary["rt_cor"][0]), self.rt_tolerance)
        self.assertLess(abs(90 - dat.summary["sd_cor"][0]), self.sd_tolerance)
        self.assertLess(abs(0.3 - dat.summary["per_err"][0]), self.err_tolerance)
        self.assertLess(abs(483 - dat.summary["rt_cor"][1]), self.rt_tolerance)
        self.assertLess(abs(103 - dat.summary["sd_cor"][1]), self.sd_tolerance)
        self.assertLess(abs(2.2 - dat.summary["per_err"][1]), self.err_tolerance)

    def test_dmcsim5(self):
        """
        Simulation 3 (Figure 5)
        amp = 20, tau = 90, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = DmcSim(tau=90)

        self.assertLess(abs(420 - dat.summary["rt_cor"][0]), self.rt_tolerance)
        self.assertLess(abs(96 - dat.summary["sd_cor"][0]), self.sd_tolerance)
        self.assertLess(abs(0.3 - dat.summary["per_err"][0]), self.err_tolerance)
        self.assertLess(abs(477 - dat.summary["rt_cor"][1]), self.rt_tolerance)
        self.assertLess(abs(96 - dat.summary["sd_cor"][1]), self.sd_tolerance)
        self.assertLess(abs(2.4 - dat.summary["per_err"][1]), self.err_tolerance)

    def test_dmcsim6(self):
        """
        Simulation 3 (Figure 5)
        amp = 20, tau = 90, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = DmcSim(tau=90, full_data=False)

        self.assertLess(abs(420 - dat.summary["rt_cor"][0]), self.rt_tolerance)
        self.assertLess(abs(96 - dat.summary["sd_cor"][0]), self.sd_tolerance)
        self.assertLess(abs(0.3 - dat.summary["per_err"][0]), self.err_tolerance)
        self.assertLess(abs(477 - dat.summary["rt_cor"][1]), self.rt_tolerance)
        self.assertLess(abs(96 - dat.summary["sd_cor"][1]), self.sd_tolerance)
        self.assertLess(abs(2.4 - dat.summary["per_err"][1]), self.err_tolerance)

    def test_dmcsim7(self):
        """
        Simulation 4 (Figure 6)
        amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = DmcSim(tau=30, var_sp=True)

        self.assertLess(abs(436 - dat.summary["rt_cor"][0]), self.rt_tolerance)
        self.assertLess(abs(116 - dat.summary["sd_cor"][0]), self.sd_tolerance)
        self.assertLess(abs(1.7 - dat.summary["per_err"][0]), self.err_tolerance)
        self.assertLess(abs(452 - dat.summary["rt_cor"][1]), self.rt_tolerance)
        self.assertLess(abs(101 - dat.summary["sd_cor"][1]), self.sd_tolerance)
        self.assertLess(abs(6.9 - dat.summary["per_err"][1]), self.err_tolerance)

    def test_dmcsim8(self):
        """
        Simulation 4 (Figure 6)
        amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = DmcSim(tau=30, full_data=False, var_sp=True)

        self.assertLess(abs(436 - dat.summary["rt_cor"][0]), self.rt_tolerance)
        self.assertLess(abs(116 - dat.summary["sd_cor"][0]), self.sd_tolerance)
        self.assertLess(abs(1.7 - dat.summary["per_err"][0]), self.err_tolerance)
        self.assertLess(abs(452 - dat.summary["rt_cor"][1]), self.rt_tolerance)
        self.assertLess(abs(101 - dat.summary["sd_cor"][1]), self.sd_tolerance)
        self.assertLess(abs(6.9 - dat.summary["per_err"][1]), self.err_tolerance)

    # def test_dmcsim9(self):
    #     """
    #     Simulation 5 (Figure 7)
    #     amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75,
    #     resMean = 300, resSD = 30
    #     """
    #     dat = DmcSim(tau=30, var_dr=True, plt_figs=False)

    #     self.assertLess(abs(500 - dat.summary["rt_cor"][0]), self.rt_tolerance)
    #     self.assertLess(abs(175 - dat.summary["sd_c"][0]), self.sd_tolerance)
    #     self.assertLess(abs(12.1 - dat.summary["per_err"][0]), self.err_tolerance)
    #     self.assertLess(abs(522 - dat.summary["rt_cor"][1]), self.rt_tolerance)
    #     self.assertLess(abs(164 - dat.summary["sd_c"][1]), self.sd_tolerance)
    #     self.assertLess(abs(13.9 - dat.summary["per_err"][1]), self.err_tolerance)

    def test_dmcsim9(self):
        """
        Simulation 5 (Figure 7)
        amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = DmcSim(tau=30, full_data=False, var_dr=True)

        self.assertLess(abs(477 - dat.summary["rt_cor"][0]), self.rt_tolerance)
        self.assertLess(abs(145 - dat.summary["sd_cor"][0]), self.sd_tolerance)
        self.assertLess(abs(3.1 - dat.summary["per_err"][0]), self.err_tolerance)
        self.assertLess(abs(494 - dat.summary["rt_cor"][1]), self.rt_tolerance)
        self.assertLess(abs(134 - dat.summary["sd_cor"][1]), self.sd_tolerance)
        self.assertLess(abs(4.1 - dat.summary["per_err"][1]), self.err_tolerance)

    # just check plot code runs
    def test_dmcsim10(self):
        dat = DmcSim(tau=30)
        try:
            dat.plot()
            dat.plot_activation()
            dat.plot_trials()
            dat.plot_pdf()
            dat.plot_cdf()
            dat.plot_delta()
            dat.plot_rt_correct()
            dat.plot_rt_error()
            dat.plot_er()
            dat.plot_rt_error()
        except:
            self.fail("*.plot() raised ExceptionType!")


if __name__ == "__main__":
    unittest.main()