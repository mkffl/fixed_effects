import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import linalg
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model.base import center_data
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class FixedEffectsEstimator(BaseEstimator, RegressorMixin):
	def __init__(self, copy_X=True, n_jobs=1):
		self.fit_intercept = True
		self.copy_X = True
		self.n_jobs = n_jobs
		self.normalize = False

	def _set_intercept(self, X_mean, y_mean, X_std):
		"""Set the intercept_
		"""
		if self.fit_intercept:
			self.coef_ = self.coef_ / X_std
			self.intercept_ = y_mean - np.dot(X_mean, self.coef_.T)
		else:
			self.intercept_ = 0.

	def fit(self, X, gps, y):
		###################
		### Check types ###
		###################
		X, y = check_X_y(X, y)
		if sp.issparse(X):
			X = X.todense()
			y = y.todense()

		# prepare group format
		if np.any(np.isnan(gps)):
			raise ValueError("groups include NaNs. Remove them and try again.")
		try:
			gps = np.array(gps).astype(int)
		except ValueError as e:
			raise ValueError("can't convert groups to dtype integer. Check if there's any string or infinite value")
		#self.gps_ = gps

		################################
		### Demean using a dataframe ###
		################################
		#************ Equation to estimate ****************
		#       _    =                 _    =         
		#yit  - yi  + y  =  a + (xit  - xi  + x)b + noise *
		#**************************************************
		#http://www.stata.com/support/faqs/statistics/intercept-in-fixed-effects-model/

		df = pd.DataFrame(X)
		cols_to_use = list(df)
		df["y"] = y
		df["gps"] = gps
		cols_demeaned = np.array(list())
		len_df = df.shape[0]

		for col in cols_to_use + ["y"]:
			average_transform = df.groupby("gps")[col].transform('mean')
			grand_average_transform = [df[col].mean()]*len_df
			col_demeaned = df[col].values - average_transform.values + grand_average_transform
			col_demeaned = np.reshape(col_demeaned, (-1, 1)) # -1 asks numpy to infer shape of original array
			if len(cols_demeaned) < 1:
				cols_demeaned = col_demeaned
			else:
				cols_demeaned = np.concatenate([cols_demeaned, col_demeaned], axis = 1)
		
		self.X_original = X
		self.y_original = y
		self.X_demeaned = cols_demeaned[:, :-1]
		self.y_demeaned = cols_demeaned[:, -1] # y is the last column of cols_demeaned
		X = cols_demeaned[:, :-1]
		y = cols_demeaned[:, -1]

		###########################
		### Regress using scipy ###
		###########################
		### Taken from scikit LinearRegression() class

		X, y, X_mean, y_mean, X_std = self._center_data(
			X, y, self.fit_intercept, self.normalize, self.copy_X,
			sample_weight=None)

		self.coef_, self._residues, self.rank_, self.singular_ = \
			linalg.lstsq(X, y)
		self.coef_ = self.coef_.T

		if y.ndim == 1:
			self.coef_ = np.ravel(self.coef_)
		self._set_intercept(X_mean, y_mean, X_std)

		#########################################
		### Estimates fixed effect intercepts ###
		#########################################
		# From the estimates a and b, estimates ui of alpha_i are obtained as ui = yi_bar − a − xi_bar*b
		# http://www.stata.com/manuals13/xtxtreg.pdf (p.25)

		ybar = df.groupby("gps")["y"].mean() # yi_bar
		mean_frame = df.groupby("gps")[cols_to_use].mean()
		x_dot_b = np.dot(mean_frame, self.coef_) # xi_bar*b
		a = np.array([self.intercept_]*ybar.shape[0])
		u = ybar - self.intercept_ - x_dot_b #
		u = pd.DataFrame(u, index = ybar.index)
		u = u.reset_index(drop = False)
		u.columns = list(u.columns[:-1])+['gps_coef']
		self.u_ = u

		# Return the classifier
		return self

	_center_data = staticmethod(center_data)


	def predict(self, X, i):
		check_is_fitted(self, "coef_")
		X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
		if sp.issparse(X):
			X = X.todense()
		if sp.issparse(i):
			i = i.todense()
		
		i_requested_unique = np.unique(i)
		i_available_unique = self.u_.gps
		for i_requested in i_requested_unique:
			if i_requested not in i_available_unique:
				raise Exception("Data to fit include unseen groups, e.g. %s" % i_requested)

		i_intercepts = pd.DataFrame(i, columns = ['gps_requested'])
		i_intercepts = pd.merge(i_intercepts, self.u_, left_on = 'gps_requested', right_on = 'gps')
		i_intercepts = i_intercepts["gps_coef"].values
		a_intercept = [self.intercept_]*X.shape[0]
		try:
			res_dot =  np.dot(X, self.coef_)
		except ValueError:
			txt = "Dot product failed. If X is a 1d array, reshape using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample."
			raise ValueError(txt)
		res = res_dot + a_intercept + i_intercepts
		return res








