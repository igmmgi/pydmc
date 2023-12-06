"""Basic unittests for pydmc.sim"""
import unittest
import pydmc


class DMCTestCaseDmcSim(unittest.TestCase):
    """Tests for Sim()"""

    rt_tolerance = 5  # ms
    sd_tolerance = 5  # ms
    err_tolerance = 0.5  # %

    def test_sim1(self):
        """
        Simulation 1 (Figure 3)
        amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = pydmc.Sim(pydmc.Prms(tau=30))

        self.assertLess(abs(440 - dat.summary["rt_cor"].iloc[0]), self.rt_tolerance)
        self.assertLess(abs(106 - dat.summary["sd_cor"].iloc[0]), self.sd_tolerance)
        self.assertLess(abs(0.7 - dat.summary["per_err"].iloc[0]), self.err_tolerance)
        self.assertLess(abs(458 - dat.summary["rt_cor"].iloc[1]), self.rt_tolerance)
        self.assertLess(abs(95 - dat.summary["sd_cor"].iloc[1]), self.sd_tolerance)
        self.assertLess(abs(1.4 - dat.summary["per_err"].iloc[1]), self.err_tolerance)

    def test_sim2(self):
        """
        Simulation 2 (Figure 4)
        amp = 20, tau = 150, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = pydmc.Sim(pydmc.Prms(tau=150))

        self.assertLess(abs(422 - dat.summary["rt_cor"].iloc[0]), self.rt_tolerance)
        self.assertLess(abs(90 - dat.summary["sd_cor"].iloc[0]), self.sd_tolerance)
        self.assertLess(abs(0.3 - dat.summary["per_err"].iloc[0]), self.err_tolerance)
        self.assertLess(abs(483 - dat.summary["rt_cor"].iloc[1]), self.rt_tolerance)
        self.assertLess(abs(103 - dat.summary["sd_cor"].iloc[1]), self.sd_tolerance)
        self.assertLess(abs(2.2 - dat.summary["per_err"].iloc[1]), self.err_tolerance)

    def test_sim3(self):
        """
        Simulation 3 (Figure 4)
        amp = 20, tau = 150, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = pydmc.Sim(pydmc.Prms(tau=150))

        self.assertLess(abs(422 - dat.summary["rt_cor"].iloc[0]), self.rt_tolerance)
        self.assertLess(abs(90 - dat.summary["sd_cor"].iloc[0]), self.sd_tolerance)
        self.assertLess(abs(0.3 - dat.summary["per_err"].iloc[0]), self.err_tolerance)
        self.assertLess(abs(483 - dat.summary["rt_cor"].iloc[1]), self.rt_tolerance)
        self.assertLess(abs(103 - dat.summary["sd_cor"].iloc[1]), self.sd_tolerance)
        self.assertLess(abs(2.2 - dat.summary["per_err"].iloc[1]), self.err_tolerance)

    def test_sim4(self):
        """
        Simulation 4 (Figure 5)
        amp = 20, tau = 90, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = pydmc.Sim(pydmc.Prms(tau=90))

        self.assertLess(abs(420 - dat.summary["rt_cor"].iloc[0]), self.rt_tolerance)
        self.assertLess(abs(96 - dat.summary["sd_cor"].iloc[0]), self.sd_tolerance)
        self.assertLess(abs(0.3 - dat.summary["per_err"].iloc[0]), self.err_tolerance)
        self.assertLess(abs(477 - dat.summary["rt_cor"].iloc[1]), self.rt_tolerance)
        self.assertLess(abs(96 - dat.summary["sd_cor"].iloc[1]), self.sd_tolerance)
        self.assertLess(abs(2.4 - dat.summary["per_err"].iloc[1]), self.err_tolerance)

    def test_sim5(self):
        """
        Simulation 5 (Figure 6)
        amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = pydmc.Sim(pydmc.Prms(tau=30, sp_dist=1))

        self.assertLess(abs(436 - dat.summary["rt_cor"].iloc[0]), self.rt_tolerance)
        self.assertLess(abs(116 - dat.summary["sd_cor"].iloc[0]), self.sd_tolerance)
        self.assertLess(abs(1.7 - dat.summary["per_err"].iloc[0]), self.err_tolerance)
        self.assertLess(abs(452 - dat.summary["rt_cor"].iloc[1]), self.rt_tolerance)
        self.assertLess(abs(101 - dat.summary["sd_cor"].iloc[1]), self.sd_tolerance)
        self.assertLess(abs(6.9 - dat.summary["per_err"].iloc[1]), self.err_tolerance)

    def test_sim6(self):
        """
        Simulation 6 (Figure 7)
        amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75,
        res_mean = 300, res_sd = 30
        """
        dat = pydmc.Sim(pydmc.Prms(tau=30, dr_dist=1))

        self.assertLess(abs(477 - dat.summary["rt_cor"].iloc[0]), self.rt_tolerance)
        self.assertLess(abs(145 - dat.summary["sd_cor"].iloc[0]), self.sd_tolerance)
        self.assertLess(abs(3.1 - dat.summary["per_err"].iloc[0]), self.err_tolerance)
        self.assertLess(abs(494 - dat.summary["rt_cor"].iloc[1]), self.rt_tolerance)
        self.assertLess(abs(134 - dat.summary["sd_cor"].iloc[1]), self.sd_tolerance)
        self.assertLess(abs(4.1 - dat.summary["per_err"].iloc[1]), self.err_tolerance)

    # just check plot code runs
    def test_sim7(self):
        dat = pydmc.Sim(pydmc.Prms(tau=30))
        self.assertTrue(dat, type(dat) == pydmc.Sim)
        try:
            pydmc.Plot(dat).summary()
            pydmc.Plot(dat).activation()
            pydmc.Plot(dat).trials()
            pydmc.Plot(dat).pdf()
            pydmc.Plot(dat).cdf()
            pydmc.Plot(dat).delta()
            pydmc.Plot(dat).rt_correct()
            pydmc.Plot(dat).rt_error()
            pydmc.Plot(dat).er()
            pydmc.Plot(dat).rt_error()
        except BaseException as error:
            print("Error {}".format(error))


class DMCTestCaseDmcOb(unittest.TestCase):
    """Tests for Ob()"""

    def test_ob1(self):
        """Flanker data check."""
        dat_flanker = pydmc.Ob(pydmc.flanker_data())

        self.assertEqual(dat_flanker.data.RT.iloc[0], 601.657182)
        self.assertEqual(dat_flanker.delta.mean_comp[0], 316.69324347500003)
        self.assertEqual(dat_flanker.caf.effect.iloc[0], 17.63034759358289)

    def test_ob2(self):
        """Simon data check."""

        dat_simon = pydmc.Ob(pydmc.simon_data())

        self.assertEqual(dat_simon.data.RT.iloc[0], 451.400105)
        self.assertEqual(dat_simon.delta.mean_comp[0], 304.90375670624996)
        self.assertEqual(dat_simon.caf.effect.iloc[0], 20.73306595365419)

    # just check plot code runs
    def test_ob3(self):
        dat_flanker = pydmc.Ob(pydmc.flanker_data())
        self.assertTrue(dat_flanker, isinstance(dat_flanker, pydmc.Ob))
        try:
            pydmc.Plot(dat_flanker).summary()
            pydmc.Plot(dat_flanker).pdf()
            pydmc.Plot(dat_flanker).cdf()
            pydmc.Plot(dat_flanker).delta()
            pydmc.Plot(dat_flanker).rt_correct()
            pydmc.Plot(dat_flanker).rt_error()
            pydmc.Plot(dat_flanker).er()
            pydmc.Plot(dat_flanker).rt_error()
        except BaseException as error:
            print("Error {}".format(error))

    # just check plot code runs
    def test_ob4(self):
        dat_simon = pydmc.Ob(pydmc.simon_data())
        self.assertTrue(dat_simon, isinstance(dat_simon, pydmc.Ob))
        try:
            pydmc.Plot(dat_simon).summary()
            pydmc.Plot(dat_simon).pdf()
            pydmc.Plot(dat_simon).cdf()
            pydmc.Plot(dat_simon).delta()
            pydmc.Plot(dat_simon).rt_correct()
            pydmc.Plot(dat_simon).rt_error()
            pydmc.Plot(dat_simon).er()
            pydmc.Plot(dat_simon).rt_error()
        except BaseException as error:
            print("Error {}".format(error))


if __name__ == "__main__":
    unittest.main()
