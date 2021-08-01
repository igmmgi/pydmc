""" Basic unittests for DMC """
import unittest
import pydmc


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
        dat = pydmc.Sim(pydmc.Prms(tau=30))

        self.assertLess(abs(440 - dat.summary["rt_cor"][0]), self.rt_tolerance)
        self.assertLess(abs(106 - dat.summary["sd_cor"][0]), self.sd_tolerance)
        self.assertLess(abs(0.7 - dat.summary["per_err"][0]), self.err_tolerance)
        self.assertLess(abs(458 - dat.summary["rt_cor"][1]), self.rt_tolerance)
        self.assertLess(abs(95 - dat.summary["sd_cor"][1]), self.sd_tolerance)
        self.assertLess(abs(1.4 - dat.summary["per_err"][1]), self.err_tolerance)

    def test_dmcsim2(self):
        """
        Simulation 2 (Figure 4)
        amp = 20, tau = 150, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = pydmc.Sim(pydmc.Prms(tau=150))

        self.assertLess(abs(422 - dat.summary["rt_cor"][0]), self.rt_tolerance)
        self.assertLess(abs(90 - dat.summary["sd_cor"][0]), self.sd_tolerance)
        self.assertLess(abs(0.3 - dat.summary["per_err"][0]), self.err_tolerance)
        self.assertLess(abs(483 - dat.summary["rt_cor"][1]), self.rt_tolerance)
        self.assertLess(abs(103 - dat.summary["sd_cor"][1]), self.sd_tolerance)
        self.assertLess(abs(2.2 - dat.summary["per_err"][1]), self.err_tolerance)

    def test_dmcsim3(self):
        """
        Simulation 3 (Figure 4)
        amp = 20, tau = 150, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = pydmc.Sim(pydmc.Prms(tau=150))

        self.assertLess(abs(422 - dat.summary["rt_cor"][0]), self.rt_tolerance)
        self.assertLess(abs(90 - dat.summary["sd_cor"][0]), self.sd_tolerance)
        self.assertLess(abs(0.3 - dat.summary["per_err"][0]), self.err_tolerance)
        self.assertLess(abs(483 - dat.summary["rt_cor"][1]), self.rt_tolerance)
        self.assertLess(abs(103 - dat.summary["sd_cor"][1]), self.sd_tolerance)
        self.assertLess(abs(2.2 - dat.summary["per_err"][1]), self.err_tolerance)

    def test_dmcsim4(self):
        """
        Simulation 4 (Figure 5)
        amp = 20, tau = 90, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = pydmc.Sim(pydmc.Prms(tau=90))

        self.assertLess(abs(420 - dat.summary["rt_cor"][0]), self.rt_tolerance)
        self.assertLess(abs(96 - dat.summary["sd_cor"][0]), self.sd_tolerance)
        self.assertLess(abs(0.3 - dat.summary["per_err"][0]), self.err_tolerance)
        self.assertLess(abs(477 - dat.summary["rt_cor"][1]), self.rt_tolerance)
        self.assertLess(abs(96 - dat.summary["sd_cor"][1]), self.sd_tolerance)
        self.assertLess(abs(2.4 - dat.summary["per_err"][1]), self.err_tolerance)

    def test_dmcsim5(self):
        """
        Simulation 5 (Figure 6)
        amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = pydmc.Sim(pydmc.Prms(tau=30, sp_dist=1))

        self.assertLess(abs(436 - dat.summary["rt_cor"][0]), self.rt_tolerance)
        self.assertLess(abs(116 - dat.summary["sd_cor"][0]), self.sd_tolerance)
        self.assertLess(abs(1.7 - dat.summary["per_err"][0]), self.err_tolerance)
        self.assertLess(abs(452 - dat.summary["rt_cor"][1]), self.rt_tolerance)
        self.assertLess(abs(101 - dat.summary["sd_cor"][1]), self.sd_tolerance)
        self.assertLess(abs(6.9 - dat.summary["per_err"][1]), self.err_tolerance)

    def test_dmcsim6(self):
        """
        Simulation 6 (Figure 7)
        amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = pydmc.Sim(pydmc.Prms(tau=30, dr_dist=1))

        self.assertLess(abs(477 - dat.summary["rt_cor"][0]), self.rt_tolerance)
        self.assertLess(abs(145 - dat.summary["sd_cor"][0]), self.sd_tolerance)
        self.assertLess(abs(3.1 - dat.summary["per_err"][0]), self.err_tolerance)
        self.assertLess(abs(494 - dat.summary["rt_cor"][1]), self.rt_tolerance)
        self.assertLess(abs(134 - dat.summary["sd_cor"][1]), self.sd_tolerance)
        self.assertLess(abs(4.1 - dat.summary["per_err"][1]), self.err_tolerance)

    # just check plot code runs
    def test_dmcsim7(self):
        dat = pydmc.Sim(pydmc.Prms(tau=30))
        self.assertTrue(dat, type(dat) == pydmc.Sim)
        try:
            pydmc.Plot(dat).plot()
            pydmc.Plot(dat).plot_activation()
            pydmc.Plot(dat).plot_trials()
            pydmc.Plot(dat).plot_pdf()
            pydmc.Plot(dat).plot_cdf()
            pydmc.Plot(dat).plot_delta()
            pydmc.Plot(dat).plot_rt_correct()
            pydmc.Plot(dat).plot_rt_error()
            pydmc.Plot(dat).plot_er()
            pydmc.Plot(dat).plot_rt_error()
        except BaseException as error:
            print("Error {}".format(error))


if __name__ == "__main__":
    unittest.main()
