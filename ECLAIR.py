import numpy as np
import pandas as pd
import time, sys

import itertools
from collections import defaultdict
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

#------------------------------------------------------------------------------------------------
#--------------------------------------- HELPER FUNCTIONS ---------------------------------------
#------------------------------------------------------------------------------------------------

def rolling_row_window(a, row_window):
    '''Join a number of rows (row_window) together into single rows over a rolling window.
    i.e. a =  array([[ nan,  nan,  nan],
					 [ nan,  nan,  nan],
					 [ nan,  nan,  nan],
					 [  0.,   1.,   2.],
					 [  3.,   4.,   5.],
					 [  6.,   7.,   8.],
					 [  9.,  10.,  11.],
		row_window = 3
		
		return: array([[ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
					   [ nan,  nan,  nan,  nan,  nan,  nan,   0.,   1.,   2.],
					   [ nan,  nan,  nan,   0.,   1.,   2.,   3.,   4.,   5.],
					   [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.],
					   [  3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.]])
		
		'''
    width = a.shape[-1]
    window = width*row_window
    a = a.flatten()
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::width]


#loss functions (can't pickle lambda functions--so we do this instead)

def cv_loss_min(aucs):
	return 1-min(aucs)


def normalize(features):
	'''normalize residue features, centered at 0 and scaled by standard deviation'''
	return (features - np.nanmean(features)) / np.nanstd(features)


#------------------------------------------------------------------------------------------------
#------------------------------------------ MAIN CLASS ------------------------------------------
#------------------------------------------------------------------------------------------------

class WinClf:

	def __init__(self, base_clf, feature_names, windows={}, hyperparams={}, verbose=False, hash_windows=False):
		''' windows - dictionary containing all feature names (pandas dataframe column names) as keys, and odd, integer windows for that feature.
		If not given, window size of 1 will be used for all features
			hyperparams - dictionary of hyperparams to pass to classifier '''
		self.base_clf = base_clf
		self.windows = windows
		self.hyperparams = hyperparams
		self.feature_names = feature_names
		
		self.verbose = verbose
		
		self.win_mask = None
		self.mask_hash = None
		self.windowed_feature_names = None
		self.max_win = None
		
		self.hash_windows = hash_windows  #boolean: store hash of windowed arrays
		self.windowed_hash = {}  #store already windowed arrays
		
		self.cv_loss = cv_loss_min   #default cross-validation loss function
		self.hyperopt_trials = Trials()   #hyperopt object to store trial information (makes warm start possible for self.optimize)
		self.param_search_results = {}   #store outcome of hyperparameter search
		
		#best trial and associated loss (from loss function) from hyperparameter search
		self.best_trial = None
		self.best_loss = None
		
		self.runtimes = defaultdict(list)
	
	
	def __str__(self):
		return 'WinClf(%s\n%s)' %(str(self.base_clf), str(self.windows))
	
	
	def get_timing_report(self):
		'''Return pandas datframe of function runtime stats'''
		runtimes = {fun: sum(r) for fun, r in self.runtimes.items()}
		runcounts = {fun: len(r) for fun, r in self.runtimes.items()}
		runmeans = {fun: np.mean(r) for fun, r in self.runtimes.items()}
		
		runstats = pd.DataFrame([runcounts, runtimes, runmeans]).T
		runstats.columns = ['Count', 'Total Time', 'Mean Time']
		return runstats
	
	
	def get_search_report(self):
		'''Collect gridsearch or hyperopt results into a pandas dataframe'''
			
		priority_columns = ['min_AUC'] + sorted(self.param_search_results[0]['window_groups'].keys()) + sorted(self.param_search_results[0]['hyperparams'].keys())
		other_columns = ['trial', 'aucs', 'pred_groups', 'pred_subgroups', 'pred_points']
		
		pd_results = pd.DataFrame(columns=priority_columns+other_columns)
		
		for i, r in enumerate(self.param_search_results):
			result_dict = {'min_AUC': min(r['cv_results']['aucs'])}
			result_dict.update(r['window_groups'])
			result_dict.update(r['hyperparams'])
			result_dict['trial'] = i
			result_dict.update(r['cv_results'])
			pd_results = pd_results.append(result_dict, ignore_index=True)
		
		pd_results.slim = priority_columns
		return pd_results
	
	
	def set_window_mask(self):
		'''Set global feature mask based on provided feature windows.'''
		
		fun_start = time.time()
		
		if self.windows != {}:
			max_win = max(self.windows.values())
		else:
			max_win = 1
		
		half_win = (max_win-1)/2
		
		win_mask = []
		win_names, win_pos = [], []
		for col in self.feature_names:
			if col in self.windows and self.windows[col]>0:
				w = self.windows[col]
				win_mask.append([0 if p<(max_win-w)/2 or p>=max_win-(max_win-w)/2 else 1 for p in range(max_win)])
			elif self.windows=={}: #take 1-window of every feature
				win_mask.append([1,])
			else: #completely mask feature
				win_mask.append([0 for p in range(max_win)])
				
			#store names of features and window offsets to later compile a list of windowed-feature names
			win_names.append([col for p in range(-1*half_win, half_win+1)])
			win_pos.append([p for p in range(-1*half_win, half_win+1)])

		#align feature masks to their columns by taking transpose
		win_mask = np.array(win_mask).T

		#take mask of window positions and feataure names (same order as actual feature masking) to get a list of windowed-feature names
		win_names = list(np.ma.masked_where(win_mask==0, np.array(win_names).T).T.compressed())
		win_pos = list(np.ma.masked_where(win_mask==0, np.array(win_pos).T).T.compressed())
		windowed_feature_names = zip(win_names, win_pos)

		self.win_mask = win_mask
		self.mask_hash = hash(win_mask.tostring())
		self.windowed_feature_names = windowed_feature_names
		self.max_win = max_win
		
		self.runtimes[sys._getframe().f_code.co_name].append(time.time()-fun_start)


	def compile_windowed_features(self, df):
		'''Apply mask based on feature windows to a dataframe. Return dataframe with same number of rows,
		but only columns associated with features in mask (and +/- window) '''
		
		fun_start = time.time()
		
		if self.hash_windows:
			df_hash = hash(df.tostring())
			
			if (df_hash, self.mask_hash) in self.windowed_hash:
				self.runtimes[sys._getframe().f_code.co_name].append(time.time()-fun_start)
				return self.windowed_hash[(df_hash, self.mask_hash)]
		
		half_win = (self.max_win-1)/2
		
		#make overflow-proof feature dataframe (concat NaN buffer top and bottom)
		nanmask = np.empty((self.max_win, df.shape[1])); nanmask.fill(np.nan)
		df_masked = np.vstack([nanmask, np.array(df), nanmask])
		
		#flatten feature array over a rolling window of rows
		df_masked_flat = rolling_row_window(df_masked, self.max_win)[half_win+1:-1*(half_win+1)]
		#flatten mask and tile to equal size of feature array
		feature_mask = np.tile(self.win_mask.flatten(), (df_masked_flat.shape[0], 1))
		
		#apply mask
		masked_array = np.ma.masked_where(feature_mask==0, df_masked_flat)
		df_features = np.ma.compress_cols(masked_array)
		
		#hash_result
		if self.hash_windows:
			self.windowed_hash[(df_hash, self.mask_hash)] = df_features
		
		#function run stats
		self.runtimes[sys._getframe().f_code.co_name].append(time.time()-fun_start)
		
		return df_features



	def fit(self, X_train, y_train):
		''' X_train - list of lists of pandas dataframes, each sub-list contains dataframes for related data
			  ALL DATAFRAMES MUST HAVE SAME COLUMNS--Missing data is indicated with np.nan
			y_train - truth labels for each datapoint, organized in same general structure as X_train: list of lists of np.arrays'''
		
		fun_start = time.time()
		
		self.set_window_mask()
		
		x_dat = []
		y_dat = np.array([], dtype=np.int)
		
		start_time = time.time()
		
		
		for g, group in enumerate(X_train):
			for d, df in enumerate(group):
				
				#get features based on windows for each core feature for each data point
				df_features = self.compile_windowed_features(df)
				
				#add y-data to finished feature dataframe to keep correspondence between features and labels
				y_dat = np.append(y_dat, y_train[g][d])
				
				#append current group member data to all-data
				x_dat.append(df_features)
		
		#recompile x-data into an array (single call to vstack is WAY faster than appending to numpy arrays within the loop)
		x_dat = np.vstack(x_dat)
		
		if self.verbose: print 'Fit Prep:', time.time()-start_time
		
		#drop rows with nans in them (including their associated y-labels)
		finite_indices = np.isfinite(x_dat).all(axis=1) & np.isfinite(y_dat)
		x_dat = x_dat[finite_indices]
		y_dat = y_dat[finite_indices]

		#set hyperparms
		self.base_clf.set_params(**self.hyperparams)
		
		start_time = time.time()
		#fit
		self.base_clf.fit(x_dat, y_dat)
		if self.verbose: print 'Fitting:', time.time()-start_time
		
		self.runtimes[sys._getframe().f_code.co_name].append(time.time()-fun_start)
		
		return self
		


	def predict_proba(self, X_test):
		''' X_train - list of lists of pandas dataframes, each sub-list contains dataframes for related data (to keep in the same fold)
			  ALL DATAFRAMES MUST HAVE SAME COLUMNS--Missing data is indicated with np.nan
			y_train - truth labels for each datapoint, organized in same general structure as X_train: list of lists of np.arrays'''
		
		fun_start = time.time()
		
		self.set_window_mask()
		
		x_dat = []
		
		group_indices = []
		df_indices = []
		pos_indices = []
		
		start_time = time.time()
		
		for g, group in enumerate(X_test):
			for d, df in enumerate(group):
				
				#get features based on windows for each core feature for each data point
				df_features = self.compile_windowed_features(df)

				#add group, df, and position information to dataframe, to keep track of where each point came from
				group_indices += [g for _ in range(len(df_features))]
				df_indices += [d for _ in range(len(df_features))]
				pos_indices += [r for r in range(len(df_features))]
								
				#append current group member data to all-data
				x_dat.append(df_features)
		
		#recompile x-data into an array
		x_dat = np.vstack(x_dat)
		group_indices = np.array(group_indices)
		df_indices = np.array(df_indices)
		pos_indices = np.array(pos_indices)
		
		
		if self.verbose: print 'Pred Prep:', time.time()-start_time
		
		#drop rows with nans in them (including their associated ID info)
		finite_indices = np.isfinite(x_dat).all(axis=1)
		x_dat = x_dat[finite_indices]
		group_indices = group_indices[finite_indices]
		df_indices = df_indices[finite_indices]
		pos_indices = pos_indices[finite_indices]
		
		start_time = time.time()
		
		#run base classifier - return 1D array of probabilities of label
		if len(x_dat) > 0:
			all_pred = self.base_clf.predict_proba(x_dat)[:,1]
		
		if self.verbose: print 'Prediction:', time.time()-start_time

		start_time = time.time()

		#organize results in same hierarchy as original X_test
		results = []
		for g, group in enumerate(X_test):
			results.append([])
			for d, df in enumerate(group):
				
				#initialize all predictions to be nan
				results[-1].append(np.array([np.nan for _ in range(len(df))]))
				
				#find original indices of predicted points
				pred_indices = pos_indices[(group_indices==g) & (df_indices==d)]
				
				if len(pred_indices) == 0:
					continue
				
				#grab predictions associated with current data
				cur_pred = all_pred[:len(pred_indices)]
				#remove those predictions from the beginning of the list
				all_pred = all_pred[len(pred_indices):] 
				
				#place probabilities associated with pred_indices into array, leaving nans intact for unpredicted points
				results[-1][-1][pred_indices] = cur_pred
		
		if self.verbose: print 'Pred wrapup: %s\n---------------------------' %(time.time()-start_time)
		
		self.runtimes[sys._getframe().f_code.co_name].append(time.time()-fun_start)
		
		return results


	def cross_validate(self, X_train, y_train, folds=3, normalize_predictions=False):
		''' clf - sklearn classifier implementing predict_proba function
			X_train - list of lists of pandas dataframes, each sub-list contains dataframes for related data (to keep in the same fold)
					  each dataframe contains index-related points for windowed calculations
					  ALL DATAFRAMES MUST HAVE SAME COLUMNS--Missing data is indicated with np.nan
			y_train - truth labels for each datapoint, organized in same general structure as X_train: list of lists of np.arrays

			Returns: AUC's for each fold '''
			
		aucs = []
		
		group_labels = range(len(X_train))
		pred_points = []
		pred_groups = []
		pred_subgroups = []
		
		#somewhat unusual use of labelkfold--each group is its own fold
		for train_indices, test_indices in cross_validation.LabelKFold(group_labels, folds):
			train_x = [x for i, x in enumerate(X_train) if i in train_indices]
			train_y = [y for i, y in enumerate(y_train) if i in train_indices]
			self.fit(train_x, train_y)
			
			test_x = [x for i, x in enumerate(X_train) if i in test_indices]
			test_y = [y for i, y in enumerate(y_train) if i in test_indices]
			predictions = self.predict_proba(test_x)
			
			#Option to compute AUCs on nomalized predictions across prediction groups
			if normalize_predictions:
				for group in range(len(predictions)):
					for subgroup in range(len(predictions[group])):
						predictions[group][subgroup] = normalize(predictions[group][subgroup])
					
			#book-keeping of how many groups and sub-groups were able to be predicted for
			concat_predictions = []
			group_finite = 0
			subgroup_finite = 0
			for group in predictions:
				group_has_finite = False
				for subgroup in group:
					concat_predictions.append(subgroup)
					if np.any(np.isfinite(subgroup)): 
						subgroup_finite += 1
						group_has_finite = True
				if group_has_finite:
					group_finite += 1
			
			#linearize predictions
			concat_truths = np.hstack(itertools.chain(*test_y))
			concat_predictions = np.hstack(concat_predictions)
					
			#drop truths/predictions where predicion is nan
			concat_truths = concat_truths[np.where(np.isfinite(concat_predictions))]
			concat_predictions = concat_predictions[np.where(np.isfinite(concat_predictions))]
			
			#calculate auc
			aucs.append(roc_auc_score(concat_truths, concat_predictions))
			pred_groups.append(group_finite)
			pred_subgroups.append(subgroup_finite)
			pred_points.append(len(concat_predictions))  #len actually predicted, with nans dropped

		
		return {'aucs': aucs, 'loss': self.cv_loss(aucs), 'pred_groups': pred_groups, 'pred_subgroups': pred_subgroups, 'pred_points': pred_points}
			
			
			
	def cv_wrapup(self, refit, X_train, y_train):
		'''refit esitmator based on best observed parameters (minimum loss)
		or clear estimator to save memory'''
		
		self.best_trial = min(self.param_search_results, key=lambda t: t['cv_results']['loss'])
		self.windows = self.best_trial['windows']
		self.hyperparams = self.best_trial['hyperparams']
		self.best_loss = self.best_trial['cv_results']['loss']
		
		if refit:
			self.fit(X_train, y_train)
		else:
			# clear latest run from memory with simple short run
			self.base_clf.set_params(**self.hyperparams)
			self.base_clf.fit([[0,0],[0,0]], [0,0])

	

	def grid_search(self, X_train, y_train, window_groups={}, hyperparam_space={}, folds=3, refit=False, normalize_predictions=False):
		'''window_groups = {'win_feat_group1': (set(['feat_A', 'feat_B', ...]), [1,3,5]), 'win_feat_group2': (set(...), [1,7])}
		   hyperparam_space = {'min_samples_leaf': [10,50,100], 'max_features': [0.5, 1.0], }'''
				
		gs_results = []
		
		for window_combo in itertools.product(*[w for f, w in window_groups.values()]):
			cur_window_groups = dict(zip(window_groups.keys(), window_combo))
			
			cur_windows = {}
			for win_group in cur_window_groups:
				for feature in window_groups[win_group][0]:
					cur_windows[feature] = cur_window_groups[win_group]
			
			for hp_combo in itertools.product(*hyperparam_space.values()):
				cur_hyperparams = dict(zip(hyperparam_space.keys(), hp_combo))
				
				self.windows = cur_windows
				self.hyperparams = cur_hyperparams
				
				#cross-validate
				if self.verbose: print cur_window_groups
				cv_results = self.cross_validate(X_train, y_train, folds, normalize_predictions)
				
				gs_results.append({'cv_results': cv_results, 'windows': cur_windows, 'window_groups': cur_window_groups, 'hyperparams': cur_hyperparams})
		
		self.param_search_results = gs_results
		
		#refit classifier to best hyperparameter set, otherwise, drop estimators to save memory
		self.cv_wrapup(refit, X_train, y_train)
		
	
	
	def optimize(self, X_train, y_train, window_groups={}, hyperparam_space={}, opt_algo=tpe.suggest, iterations=10, early_stop=False, folds=3, refit=False, warm_start=False, normalize_predictions=False):
		'''
			window_groups = {'win_feat_group1': (type1_features, 1+hp.quniform('win_feat_group1', 0, 10, 2)), 
							'win_feat_group2': (type2_features, 1+hp.quniform('win_feat_group2', 0, 10, 2))}
		   
			hyperparam_space = {'min_samples_leaf': hp.quniform('min_samples_leaf',1,200,1),
							   'max_features': hp.uniform('max_features', 0,1)}
							   
			opt_algo: algorithm to choose subsequent hyperparameters (hyperopt.tpe.suggest, hyperopt.random.suggest)
		'''
		
		
		def objectiveFun(params):
			
			cur_window_groups = params['cur_window_groups']
			cur_hyperparams = params['cur_hyperparams']
			
			cur_windows = {}
			for win_group in cur_window_groups:
				for feature in window_groups[win_group][0]:
					cur_windows[feature] = int(cur_window_groups[win_group])
			
			self.windows = cur_windows
			self.hyperparams = cur_hyperparams
			
			#cross-validate
			cv_results = self.cross_validate(X_train, y_train, folds, normalize_predictions)
			
			return {
				'loss': cv_results['loss'],  #value to minimize
				'status': STATUS_OK,  #required 
				'cv_results': cv_results,
				'windows': cur_windows,
				'window_groups': cur_window_groups,
				'hyperparams': cur_hyperparams,
				}
		
		#--------------------------------------------------------------
		
		if not warm_start:
			self.hyperopt_trials = Trials() #object to save trial information
		
		
		for iterat in range(iterations):
		
			#run optimizer
			best = fmin(objectiveFun,
				space={'cur_window_groups': {wg: window_groups[wg][1] for wg in window_groups},
					   'cur_hyperparams': hyperparam_space},
				algo=opt_algo,
				max_evals=len(self.hyperopt_trials.trials)+1,
				trials=self.hyperopt_trials)
			
			if early_stop != False:
				
				if len(self.hyperopt_trials.trials) < early_stop+1:
					continue
				
				best_losses = np.minimum.accumulate([t['result']['loss'] for t in self.hyperopt_trials.trials])
				if best_losses[-1] == best_losses[-1*(early_stop+1)]:
					break
				
		
		#collect results into pandas dataframe
		self.param_search_results = [t['result'] for t in self.hyperopt_trials.trials]
		
		#refit classifier to best hyperparameter set, otherwise, drop estimators to save memory
		self.cv_wrapup(refit, X_train, y_train)

			
		
