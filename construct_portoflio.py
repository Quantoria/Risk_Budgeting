"""
    Copyright 2017 Victoria Xie

    This file is part of Macro_Nowcast.

    Risk_Budgeting is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Risk_Budgeting is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

"""
This file constructs a strategic ERP portfolio and compare with an equal weighted portfolio.
"""


# risk budgeting approach optimisation object function
def obj_fun(x, p_cov, rb):
	return np.sum((x*np.dot(p_cov, x)/np.dot(x.transpose(), np.dot(p_cov, x))-rb)**2)


# constraint on sum of weights equal to one
def cons_sum_weight(x):
	return np.sum(x)-1.0


# constraint on weight larger than zero
def cons_long_only_weight(x):
	return x


# calculate risk budgeting portfolio weight give risk budget
def rb_p_weights(asset_rets, rb):
	# number of ARP series
	num_arp = asset_rets.shape[1]
	# covariance matrix of asset returns
	p_cov = asset_rets.cov()
	# initial weights
	w0 = 1.0 * np.ones((num_arp, 1)) / num_arp
	# constraints
	cons = ({'type': 'eq', 'fun': cons_sum_weight}, {'type': 'ineq', 'fun': cons_long_only_weight})
	# portfolio optimisation
	return minimize(obj_fun, w0, args=(p_cov, rb), method='SLSQP', constraints=cons)


if __name__ == "__main__":
	# 1. Load ARP data
	rf_data = pd.read_excel("data/data_rp.xlsx", "RF")  # load daily risk free rate data
	arp_data = pd.read_excel("data/data_rp.xlsx", "RP")  # load daily risk premia data
	rf_data = rf_data[1:]
	arp_data = arp_data[1:]
	rf_data = rf_data.apply(pd.to_numeric)
	arp_data = arp_data.apply(pd.to_numeric)

	# 2. Calculate ARP excess returns
	arp_rets = (np.log(arp_data) - np.log(arp_data.shift(1)))[1:]
	arp_rets = arp_rets.sub(rf_data.squeeze()/252, axis='index')

	# 3. Construct risk budgeting portfolio
	# portfolio dates
	p_dates = arp_rets.index[arp_rets.index >= '2005-01-03']
	# previous month
	pre_mth = 12

	# initialise portfolio weights matrix
	w = pd.DataFrame(index=p_dates, columns=arp_rets.columns)
	# initialise portfolio return matrix
	p_rets = pd.DataFrame(index=p_dates, columns=['Risk Parity'])

	for t in p_dates:

		# construct risk budgeting portfolio and re-balance on monthly basis
		if t.month==pre_mth:
			# keep the same portfolio weights within the month
			w.ix[t] = w.iloc[w.index.get_loc(t)-1]
		else:
			# update the value of the previous month record
			pre_mth = t.month
			# re-balance the portfolio at the start of the month
			w.ix[t] = rb_p_weights(arp_rets[arp_rets.index < t], 1.0/num_arp).x

		# calculate risk budgeting portfolio returns
		p_rets.ix[t] = np.sum(w.ix[t] * arp_rets.ix[t])

	# 4. Construct equal weighted portfolio
	ew_rets = pd.DataFrame(np.sum(1.0*arp_rets[arp_rets.index>=p_dates[0]]/num_arp, axis=1), columns=['Equal Weighted'])

	# 5. Plot the portfolio cumulative returns
	p_cumrets = (p_rets + 1).cumprod()
	ew_cumrets = (ew_rets + 1).cumprod()

	pd.concat([p_cumrets, ew_cumrets], axis=1).plot()
	plt.show()
