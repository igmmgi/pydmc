# import numpy as np
# from scipy.optimize import minimize
# from pydmc.dmcob import DmcOb, flankerDataRaw
# from pydmc.dmcsim import DmcSim
#
#
# class DmcFit:
#     def __init__(
#         self,
#         res_ob,
#         n_trls=100000,
#         start_vals=None,
#         min_vals=None,
#         max_vals=None,
#         fixed_fit=None,
#         n_delta=19,
#         n_caf=5,
#         var_sp=True,
#     ):
#         """
#         Parameters
#         ----------
#         """
#         self.res_ob = res_ob
#         self.n_trls = n_trls
#         self.start_vals = start_vals
#         self.min_vals = min_vals
#         self.max_vals = min_vals
#         self.fixed_fit = fixed_fit
#         self.n_delta = n_delta
#         self.n_caf = n_caf
#         self.var_sp = var_sp
#         start_vals = {
#             "amp": 20,
#             "tau": 150,
#             "drc": 0.5,
#             "bnds": 75,
#             "resMean": 300,
#             "resSD": 30,
#             "aaShape": 2,
#             "spShape": 3,
#             "sigm": 4,
#         }
#         if self.start_vals is None:
#             self.start_vals = start_vals
#         else:
#             start_vals.update(self.start_vals)
#             self.start_vals = start_vals
#         min_vals = {
#             "amp": 0,
#             "tau": 5,
#             "drc": 0.1,
#             "bnds": 20,
#             "resMean": 200,
#             "resSD": 5,
#             "aaShape": 1,
#             "spShape": 2,
#             "sigm": 1,
#         }
#         if self.min_vals is None:
#             self.min_vals = min_vals
#         else:
#             min_vals.update(self.min_vals)
#             self.min_vals = min_vals
#         max_vals = {
#             "amp": 40,
#             "tau": 300,
#             "drc": 1.0,
#             "bnds": 150,
#             "resMean": 800,
#             "resSD": 100,
#             "aaShape": 3,
#             "spShape": 4,
#             "sigm": 10,
#         }
#         if self.max_vals is None:
#             self.max_vals = max_vals
#         else:
#             max_vals.update(self.max_vals)
#             self.max_vals = max_vals
#         fixed_fit = {
#             "amp": False,
#             "tau": False,
#             "drc": False,
#             "bnds": False,
#             "resMean": False,
#             "resSD": False,
#             "aaShape": False,
#             "spShape": False,
#             "sigm": True,
#         }
#         if self.fixed_fit is None:
#             self.fixed_fit = fixed_fit
#         else:
#             fixed_fit.update(self.fixed_fit)
#             self.fixed_fit = fixed_fit
#
#         for key, value in self.fixed_fit.items():
#             if value:
#                 self.min_vals[key] = self.start_vals[key]
#                 self.max_vals[key] = self.start_vals[key]
#         print(self.min_vals, self.max_vals)
#
#     def fit_data(self):
#         self.fit = minimize(
#             self._function_to_minimise,
#             list(self.start_vals.values()),
#             (res_ob, self.n_trls, self.n_delta, self.n_caf),
#             options={
#                 "adaptive": True,
#                 "bounds": [
#                     (20, 20),
#                     (20, 20),
#                     (20, 20),
#                     (20, 20),
#                     (20, 20),
#                     (20, 20),
#                     (20, 20),
#                     (20, 20),
#                 ],
#             },
#         )
#
#     def _function_to_minimise(self, x, res_ob, n_trls, n_delta, n_caf):
#
#         # # bounds hack
#         # x = np.maximum(x, list(self.min_vals.values()))
#         # x = np.minimum(x, list(self.max_vals.values()))
#         print(x)
#
#         res_th = DmcSim(
#             amp=x[0],
#             tau=x[1],
#             drc=x[2],
#             bnds=x[3],
#             res_mean=x[4],
#             res_sd=x[5],
#             aa_shape=x[6],
#             sp_shape=x[7],
#             sigma=x[8],
#             n_trls=n_trls,
#             n_delta=n_delta,
#             n_caf=n_caf,
#             res_dist=1,
#         )
#
#         return self.calculate_cost_value_rmse(res_th, res_ob)
#
#     @staticmethod
#     def calculate_cost_value_rmse(res_th, res_ob):
#         """calculate_cost_value_rmse"""
#         n_rt = len(res_th.delta) * 2
#         n_err = len(res_th.caf)
#
#         np.sum(
#             np.sum(
#                 res_th.delta[["mean_comp", "mean_incomp"]]
#                 - res_ob.delta[["mean_comp", "mean_incomp"]]
#             )
#         )
#
#         cost_caf = np.sqrt(
#             (1 / n_err) * np.sum((res_th.caf["Error"] - res_ob.caf["Error"]) ** 2)
#         )
#         cost_rt = np.sqrt(
#             (1 / n_rt)
#             * np.sum(
#                 np.sum(
#                     res_th.delta[["mean_comp", "mean_incomp"]]
#                     - res_ob.delta[["mean_comp", "mean_incomp"]]
#                 )
#             )
#         )
#         weight_rt = n_rt / (n_rt + n_err)
#         weight_caf = (1 - weight_rt) * 1500
#
#         cost_value = (weight_caf * cost_caf) + (weight_rt + cost_rt)
#
#         if np.isnan(cost_value):
#             cost_value = np.inf
#
#         print(f"RMSE: {cost_value:.2f}")
#
#         return cost_value
#
#
# # res_th = DmcSim()
# res_ob = DmcOb(flankerDataRaw())
#
# fit = DmcFit(res_ob)
#
# fit.fit_data()
# fit.fit
