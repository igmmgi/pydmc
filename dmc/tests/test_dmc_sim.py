"""Basic unittests for expfun.mix"""
import unittest
from dmc import dmc_sim


class DMCTestCaseDMC(unittest.TestCase):
    """Tests for dmc_sim()"""
 
    rt_tolerance = 3    # ms
    sd_tolerance = 3    # ms
    err_tolerance = 0.5 # %
    
    def test_dmc_sim1(self):
        """
        Simulation 1 (Figure 3)
        amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75, 
        resMean = 300, resSD = 30
        """
        res = dmc_sim.simulation(tau = 30, plt_figs=False)
        
        self.assertLess(abs(440 - res["rtCorr"][0]), self.rt_tolerance)  
        self.assertLess(abs(106 - res["sdCorr"][0]), self.sd_tolerance)  
        self.assertLess(abs(0.7 - res["perErr"][0]), self.err_tolerance) 
        self.assertLess(abs(458 - res["rtCorr"][1]), self.rt_tolerance)   
        self.assertLess(abs(95  - res["sdCorr"][1]), self.sd_tolerance)  
        self.assertLess(abs(1.4 - res["perErr"][1]), self.err_tolerance) 

    def test_dmc_sim2(self):
        """
        Simulation 2 (Figure 4)
        amp = 20, tau = 150, mu = 0.5, sigm = 4, bnds = 75, 
        resMean = 300, resSD = 30
        """
        res = dmc_sim.simulation(tau = 150, plt_figs=False)
        
        self.assertLess(abs(422 - res["rtCorr"][0]), self.rt_tolerance)  
        self.assertLess(abs(90  - res["sdCorr"][0]), self.sd_tolerance)  
        self.assertLess(abs(0.3 - res["perErr"][0]), self.err_tolerance) 
        self.assertLess(abs(483 - res["rtCorr"][1]), self.rt_tolerance)   
        self.assertLess(abs(103 - res["sdCorr"][1]), self.sd_tolerance)  
        self.assertLess(abs(2.2 - res["perErr"][1]), self.err_tolerance) 

    def test_dmc_sim3(self):
        """
        Simulation 3 (Figure 5)
        amp = 20, tau = 90, mu = 0.5, sigm = 4, bnds = 75, 
        resMean = 300, resSD = 30
        """
        res = dmc_sim.simulation(tau = 90, plt_figs=False)
        
        self.assertLess(abs(420 - res["rtCorr"][0]), self.rt_tolerance)  
        self.assertLess(abs(96  - res["sdCorr"][0]), self.sd_tolerance)  
        self.assertLess(abs(0.3 - res["perErr"][0]), self.err_tolerance) 
        self.assertLess(abs(477 - res["rtCorr"][1]), self.rt_tolerance)   
        self.assertLess(abs(96  - res["sdCorr"][1]), self.sd_tolerance)  
        self.assertLess(abs(2.4 - res["perErr"][1]), self.err_tolerance) 

    def test_dmc_sim4(self):
        """
        Simulation 4 (Figure 6)
        amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75, 
        resMean = 300, resSD = 30
        """
        res = dmc_sim.simulation(tau = 30, var_sp=True, plt_figs=False)
        
        self.assertLess(abs(436 - res["rtCorr"][0]), self.rt_tolerance)  
        self.assertLess(abs(116 - res["sdCorr"][0]), self.sd_tolerance)  
        self.assertLess(abs(1.7 - res["perErr"][0]), self.err_tolerance) 
        self.assertLess(abs(452 - res["rtCorr"][1]), self.rt_tolerance)   
        self.assertLess(abs(101 - res["sdCorr"][1]), self.sd_tolerance)  
        self.assertLess(abs(6.9 - res["perErr"][1]), self.err_tolerance) 

    def test_dmc_sim5(self):
        """
        Simulation 5 (Figure 7)
        amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75, 
        resMean = 300, resSD = 30
        """
        res = dmc_sim.simulation(tau = 30, var_dr=True, plt_figs=False)
        
        self.assertLess(abs(500  - res["rtCorr"][0]), self.rt_tolerance)  
        self.assertLess(abs(175  - res["sdCorr"][0]), self.sd_tolerance)  
        self.assertLess(abs(12.1 - res["perErr"][0]), self.err_tolerance) 
        self.assertLess(abs(522  - res["rtCorr"][1]), self.rt_tolerance)   
        self.assertLess(abs(164  - res["sdCorr"][1]), self.sd_tolerance)  
        self.assertLess(abs(13.9 - res["perErr"][1]), self.err_tolerance) 
        
if __name__ == "__main__":
    unittest.main()
