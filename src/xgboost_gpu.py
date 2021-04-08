'''
Copyright 2021 Guilherme Oliveira
SPDX-License-Identifier: Apache-2.0
========================================================================================================================
Author: Guilherme Oliveira
Date: july 06, 2020
Contact: gmmoliveira1@gmail.com
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
========================================================================================================================
This script implements functions to facilitate the execution of the XGBoost algorithm on multiple GPUs on a single-
machine.
========================================================================================================================
'''
from xgboost.dask import DaskDMatrix, train as dask_xgboost_train, predict as dask_xgboost_predict
from dask.dataframe import from_array, from_pandas
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import numpy as np
import pandas as pd


def train_xgboost_gpu(
		X, y,
		data_chunksize=None,
		n_gpus=None, n_threads_per_gpu=1,
		params=None,
		xgboost_model=None,
		gpu_cluster=None, client=None
):
	'''
	Trains a XGBoost model on the GPU.

	:param X: a 2D matrix object of either type numpy ndarray or pandas DataFrame;
	:param y: a 1D array of one of the following types: numpy ndarray, pandas Series or pandas DataFrame;

	:param data_chunksize: number of rows to partition input data (both X and y simultaneously) to split among multiple
		GPU devices. Default value None splits evenly among devices;
	:param n_gpus: number of GPUs to be used. Default value None selects all available devices.
	:param n_threads_per_gpu: number of threads per GPU;
	:param params: xgboost trainning params as a python dict, refer to
		https://xgboost.readthedocs.io/en/latest/parameter.html
	:param xgboost_model: xgbooster object to continue training, it may be either a regular XGBoost model or a
		dask xgboost dict
	:param gpu_cluster: an existing dask cluster object to use. This param should be used if you call this method
		too many times in quick successions. Note that this function doesn't close an externally created cluster.
	:param client: an existing dask client object to use. This param should be used if you call this method
		too many times in quick successions. Note that this function doesn't close an externally created client.

	:return:
	A dictionary containing 2 keys:
		* 'booster': maps to a XGBoost model
		* 'history': maps to  another dict which informs the history of the training process, as in the following the
			examṕle: {'train': {'logloss': ['0.48253', '0.35953']}, 'eval': {'logloss': ['0.480385', '0.357756']}}}
	'''

	if gpu_cluster is None:
		local_gpus = LocalCUDACluster(n_workers=n_gpus, threads_per_worker=n_threads_per_gpu)
	else:
		local_gpus = gpu_cluster
	if client is None:
		local_dask_client = Client(local_gpus, {'verbose': 0})
	else:
		local_dask_client = client

	if data_chunksize is None:
		data_chunksize = X.shape[0] // len(local_gpus.cuda_visible_devices)
	if params is None:
		params = {
			'learning_rate': 0.3,
			'max_depth': 8,
			'objective': 'reg:squarederror',
			'verbosity': 0,
			'tree_method': 'gpu_hist'
		}

	if isinstance(X, pd.DataFrame):
		X = from_pandas(X, chunksize=data_chunksize)
	else:
		X = from_array(X, chunksize=data_chunksize)
	if isinstance(y, pd.DataFrame):
		y = from_pandas(y, chunksize=data_chunksize)
	else:
		y = from_array(y, chunksize=data_chunksize)
	dtrain = DaskDMatrix(local_dask_client, X, y)

	if type(xgboost_model) is dict:
		xgboost_model = xgboost_model['booster']

	xgb_model = dask_xgboost_train(local_dask_client, params, dtrain, num_boost_round=100, evals=[(dtrain, 'train')],
	                               xgb_model=xgboost_model)

	if client is None:
		local_dask_client.close()
	if gpu_cluster is None:
		local_gpus.close()

	return xgb_model


def predict_xgboost_gpu(
		xgb_model, X,
		data_chunksize=None,
		n_gpus=None, n_threads_per_gpu=1,
		gpu_cluster=None, client=None
):
	'''
	Predicts the output for the input features X using the 'xgb_model' running on the GPU.

	:param xgb_model: a dask XGBoost model to use for predictions
	:param X: the input features to use for predictions, must be either a numpy ndarray or a pandas DataFrame
	:param data_chunksize: chunk sizes to be used on a dask dataframe, leave the default value None for auto decision
	:param n_gpus: number of GPUs to be used. Default value None selects all available devices;
	:param n_threads_per_gpu: number of threads per GPU;
	:param gpu_cluster: an existing dask cluster object to use. This param should be used if you call this method
		too many times in quick successions. Note that this function doesn't close an externally created cluster.
	:param client: an existing dask cluster object to use. This param should be used if you call this method
		too many times in quick successions. Note that this function doesn't close an externally created client.
	:return:
		If the input features X is a pandas DataFrame, returns a array-like DataFrame of single column containing
		the predictions;

		Otherwise, if the input features X is a numpy ndarray, returns a 1D ndarray containing the predictions .
	'''
	if gpu_cluster is None:
		local_gpus = LocalCUDACluster(n_workers=n_gpus, threads_per_worker=n_threads_per_gpu)
	else:
		local_gpus = gpu_cluster
	if client is None:
		local_dask_client = Client(local_gpus)
	else:
		local_dask_client = client

	if data_chunksize is None:
		data_chunksize = X.shape[0] // len(local_gpus.cuda_visible_devices)

	if isinstance(X, pd.DataFrame):
		ndarray = False
		X = from_pandas(X, chunksize=data_chunksize)
	else:
		ndarray = True
		X = from_array(X, chunksize=data_chunksize)

	y_predicted = dask_xgboost_predict(local_dask_client, xgb_model, X)
	y_predicted = pd.DataFrame(y_predicted)

	if client is None:
		local_dask_client.close()
	if gpu_cluster is None:
		local_gpus.close()

	if ndarray:
		return y_predicted.to_numpy()
	return y_predicted


def _example():
	# the following imports are meant to be used only in the scope of this example, therefore,
	# they were placed here for performance issues regarding external modules calling this one
	from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
	from sklearn.metrics import explained_variance_score, mean_squared_error, max_error
	from os.path import exists

	base_path = ''
	if exists('../models/'):
		base_path = '../models/'

	# [WARNING]: choose carefully the below parameters according to your machine, avoiding, for example, consuming
	# more memory than what's available
	n, m = 10 ** 4, 10
	rand = np.random.Generator(np.random.PCG64())

	print('========== *** XGBoost Classification example *** ==========')
	params = {
		'learning_rate': 0.3,
		'max_depth': 8,
		'objective': 'binary:hinge',
		'verbosity': 0,
		'tree_method': 'gpu_hist'
	}
	class_proportion = 0.5
	X = rand.random(size=(n, m))
	y = np.array([1 if np.sum(X[i, :]) > class_proportion * m else 0 for i in range(X.shape[0])])
	classification_xgbmodel = train_xgboost_gpu(X, y, params=params, n_gpus=1, n_threads_per_gpu=1, xgboost_model=None)

	X = rand.random(size=(n, m))
	y = np.array([1 if np.sum(X[i, :]) > class_proportion * m else 0 for i in range(X.shape[0])])
	y_pred = predict_xgboost_gpu(classification_xgbmodel, X, n_gpus=1, n_threads_per_gpu=1)
	'''
	# my tests have shown that predicting over the GPU is much slower than over the CPU
	# to predict using the CPU instead of the GPU, use the following example code
	from xgboost import DMatrix
	y_pred = classification_xgbmodel['booster'].predict(DMatrix(pd.DataFrame(X, columns=[i for i in range(m)])))
	'''

	acc = accuracy_score(y, y_pred)
	cm = confusion_matrix(y, y_pred)
	print('accuracy: {:.2f}%'.format(acc * 100))
	print('confusion matrix:')
	print(cm)
	try:
		print('ROC AUC score: {:.2f}%'.format(roc_auc_score(y, y_pred) * 100))
	except:
		pass

	# save your model as follows
	classification_xgbmodel['booster'].save_model(base_path + 'my_classf_model001.xgbmodel')

	print('========== *** XGBoost Regression example *** ==========')
	transformation = rand.random(size=m)
	X = rand.random(size=(n, m))
	y = np.matmul(X, transformation)
	params = {
		'learning_rate': 0.3,
		'max_depth': 8,
		'objective': 'reg:squarederror',
		'verbosity': 0,
		'tree_method': 'gpu_hist'
	}
	regression_xgbmodel = train_xgboost_gpu(X, y, params=params)
	X = rand.random(size=(n, m))
	y = np.matmul(X, transformation)
	y_pred = predict_xgboost_gpu(regression_xgbmodel, X)
	'''
	# my tests have shown that predicting over the GPU is much slower than over the CPU
	# to predict using the CPU instead of the GPU, use the following example code
	from xgboost import DMatrix
	y_pred = regression_xgbmodel['booster'].predict(DMatrix(pd.DataFrame(X, columns=[i for i in range(m)])))
	'''

	vscore = explained_variance_score(y, y_pred)
	mse = mean_squared_error(y, y_pred)
	me = max_error(y, y_pred)
	print('Variance score: {:.2f}'.format(vscore))
	print('Mean squared error: {:.2f}'.format(mse))
	print('Maximum absolute error: {:.2f}'.format(me))

	# save your model as follows
	regression_xgbmodel['booster'].save_model(base_path + 'my_reg_model001.xgbmodel')


if __name__ == '__main__':
	from time import time

	t_start = time()
	_example()
	t_end = time() - t_start
	print('executed in {:.2f} seconds'.format(t_end))