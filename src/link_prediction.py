'''
Copyright 2021 Guilherme Oliveira
SPDX-License-Identifier: Apache-2.0
========================================================================================================================
Author: Guilherme Oliveira
Date: april 08, 2021
Contact: gmmoliveira1@gmail.com
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
========================================================================================================================
This script implements functions presenting how to perform link prediction on undirected unweighted/weighted graphs,
using unsupervised learning to obtain vertices embeddings, where the graph may have an extremely
imbalance between edges and non-edges.

A non-edge may be defined as: given a graph G(V, E), further, given
a pair of vertices v1 and v2 which belongs to V, a non-edge is a non-existing link between v1 and v2, that is,
the edge (v1, v2) doesn't belong to E.
========================================================================================================================
'''

import numpy as np
import networkx as nx
from nbne import train_model
from os.path import join, abspath, dirname
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from time import time
import xgboost
import tensorflow as tf
import os
import warnings
import xgboost_gpu


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')


def get_nbne_embeddings(
						graph,
						num_permutations=10,
						embedding_dimension=128,
						window_size=5,
						min_count=0,
						min_degree=0,
						workers=8,
						dtype=np.float64
						):
	'''
	Returns a 2D numpy.ndarray where the i-th row represent's the i-th node in the networkx.Graph "graph" and the
	respective associated columns represents the embeddings of vertex i. Note that this function considers the names of
	all nodes in the graph to be integers in the range [0, number of nodes in the graph], that allows for the
	abstraction of mapping each vertex to a unique position in the numpy.ndarray, creating an efficient data structure
	for later usage;
	:param graph: a networkx.Graph representing the graph to extract the embeddings from;
	:param num_permutations: a NBNE's parameter defining the number of permutations performed using the neighborhood of
		a vertex;
	:param embedding_dimension: the output number of dimensions for the embeddings of each vertex;
	:param window_size: the window size used to predict a node given it's neighbours;
	:param min_count: ignores all vertices with total frequency lower than this. Frequency here is considering the
		occurrence of a vertex as a neighbour of other vertices;
	:param min_degree: ignores all vertices with degree lower than this;
	:param workers: number of threads Gensim's training uses (it's called by the NBNE internally);
	:param dtype: the data type of the output embeddings numpy.ndarray;
	:return:
		* emb: a 2D numpy.ndarray where the i-th row represent's the i-th node in the networkx.Graph "graph" and the
		respective associated columns represents the embeddings of vertex i. This structure is shaped as follows:
		(number of vertices in the graph, embedding_dimension);
	'''
	model = train_model(
						graph,
						num_permutations=num_permutations,
						output_file=None,
						embedding_dimension=embedding_dimension,
						window_size=window_size,
						min_count=min_count,
						min_degree=min_degree,
						workers=workers
						)
	emb = np.empty(shape=(graph.number_of_nodes(), embedding_dimension), dtype=dtype)
	for word in model.wv.index2entity:
		emb[int(word), :] = model.wv.word_vec(word=word)
	return emb


def _read_graph(file_name, adj_matrix_dtype=np.float64):
	'''
	Retrieves from a graph from an "edgelist" file (check networkx documentation for more details on this specific file
	format) specified by it's name without the associated operating system path. The file will be searched in the
	"graphs" folder of this project, no error handling is performed.
	:param file_name: the raw file name, i. e., without it's path. The path is assumed to be the "graphs" folder
		inside this project on a cross-platform fashion;
	:param adj_matrix_dtype: the data type to of the output "A" matrix;
	:return:
	Returns two versions of the same undirected graph:
		* G: a networkx.Graph instance containing the file's structured data;
		* A: a numpy.ndarray containing the equivalent adjacency matrix. This structure is mean't for greater
			efficiency when generating the training data for link prediction;
	'''
	full_path = join(join(dirname(dirname(abspath(__file__))), "graphs"), file_name)
	G = nx.read_edgelist(full_path)
	A = nx.to_numpy_array(G, dtype=adj_matrix_dtype)
	return G, A


def assemble_balanced_link_prediction_data(A, emb, sampling_rate=0.5, dtype=np.float32):
	'''
	Generates class balanced training/testing data for graph "A" using it's embeddings "emb". This function
	is meant to be used with
	undirected unweighted/weighted graphs, since it only considers the upper diagonal of "A" due to efficiency purposes.
	Self loops are also ignored;

	:param A: a square 2D numpy.ndarray representing the graph's adjacency matrix;
	:param emb: a 2D numpy.ndarray where each row represents a vertex from A and the columns represents the embeddings
		matching the vertex in the same row;
	:param sampling_rate: a float in the interval (0.0, 1.0] determining the rate at which links and non-links will be
		sampled from the graph to be used in the training data produced;
	:param dtype: the data type of the produced training data;
	:return:
		* X_train: a 2D numpy.ndarray representing the training features;
		* y_train: a 1D numpy.ndarray representing the training labels/targets;
		'''
	n = A.shape[0]
	max_instances = int((((n ** 2) - n) / 2) * sampling_rate)
	zeros_idx, ones_idx = [], []
	for i in range(n):
		for j in range(i + 1, n - 1):
			if float(A[i, j]) == 0.0:
				zeros_idx.append((i, j, 0))
			else:
				ones_idx.append((i, j, 1))
	dominant_class = 1 if len(ones_idx) > len(zeros_idx) else 0
	np.random.shuffle(zeros_idx)
	np.random.shuffle(ones_idx)
	if dominant_class == 0:
		zeros_idx = zeros_idx[:len(ones_idx)]
	else:
		ones_idx = ones_idx[:len(zeros_idx)]
	if len(zeros_idx) + len(ones_idx) > max_instances:
		diff = np.ceil(np.abs(len(zeros_idx) + len(ones_idx) - max_instances) / 2)
		zeros_idx = zeros_idx[:-diff]
		ones_idx = ones_idx[:-diff]

	data = np.empty(shape=(len(zeros_idx) + len(ones_idx), 2 * emb.shape[1] + 1), dtype=dtype)
	for k, (i, j, label) in enumerate(zeros_idx + ones_idx):
		data[k, :emb.shape[1]] = emb[i, :]
		data[k, emb.shape[1]:-1] = emb[j, :]
		data[k, -1] = label
	np.random.shuffle(data)
	x = data[:, :-1]
	y = data[:, -1]

	return x, y


def print_scores_summary(ytrue, ypredicted, training_time, inference_time, model_name="Model"):
	def fmt_fixed_len(str, fixed_length=25):
		assert fixed_length >= len(str)
		n = fixed_length - len(str)
		return str + (" " * n)

	print("{} {} results {}".format("=" * 5, model_name, "=" * 5))
	print("{}{:.2f}%".format(fmt_fixed_len("Accuracy:"), accuracy_score(ytrue, ypredicted) * 100))
	print("{}{:.2f}%".format(fmt_fixed_len("ROC AUC:"), roc_auc_score(ytrue, ypredicted) * 100))
	print("{}{:.2f}%".format(fmt_fixed_len("F1:"), f1_score(ytrue, ypredicted) * 100))
	print("{}{:.2f}%".format(fmt_fixed_len("Precision:"), precision_score(ytrue, ypredicted) * 100))
	print("{}{:.2f}%".format(fmt_fixed_len("Recall:"), recall_score(ytrue, ypredicted) * 100))
	print("{} training time: {:.2f} seconds.".format(model_name, training_time))
	print("{} inference time: {:.2f} seconds.".format(model_name, inference_time))


def make_nn(input_size):
	model_inputs = tf.keras.Input(shape=input_size, dtype=tf.float32)
	#layer = tf.keras.layers.Reshape((2, -1))(model_inputs)
	half = int(input_size / 2)
	layer0 = tf.keras.layers.Reshape((1, -1))(model_inputs[:, :half])
	layer1 = tf.keras.layers.Reshape((1, -1))(model_inputs[:, half:])

	a0 = tf.keras.layers.MultiHeadAttention(num_heads=16, key_dim=16, value_dim=16, dropout=0.0)(layer0, layer0)
	a00 = tf.keras.layers.LayerNormalization()(a0 + layer0)
	a0 = tf.keras.layers.Concatenate(axis=1)([a0, a00])
	a1 = tf.keras.layers.MultiHeadAttention(num_heads=16, key_dim=16, value_dim=16, dropout=0.0)(layer1, layer1)
	a11 = tf.keras.layers.LayerNormalization()(a1 + layer1)
	a1 = tf.keras.layers.Concatenate(axis=1)([a1, a11])
	a2 = tf.keras.layers.MultiHeadAttention(num_heads=16, key_dim=16, value_dim=16, dropout=0.0)(layer0, layer1)
	a22 = tf.keras.layers.LayerNormalization()(a2 + layer0 + layer1)
	a2 = tf.keras.layers.Concatenate(axis=1)([a2, a22])
	a3 = tf.keras.layers.MultiHeadAttention(num_heads=16, key_dim=16, value_dim=16, dropout=0.0)(layer1, layer0)
	a33 = tf.keras.layers.LayerNormalization()(a3 + layer1 + layer0)
	a3 = tf.keras.layers.Concatenate(axis=1)([a3, a33])
	layer = tf.keras.layers.Concatenate(axis=1)([a0, a1, a2, a3])

	partially_encoded = tf.keras.layers.Flatten()(layer)

	layer = tf.keras.layers.Dense(128, activation="relu")(partially_encoded)
	layer = tf.keras.layers.Dense(partially_encoded.shape[-1], activation="relu")(layer)
	layer = tf.keras.layers.LayerNormalization()(layer + partially_encoded)
	layer = tf.keras.layers.Concatenate(axis=-1)([layer, partially_encoded])
	model_outputs = tf.keras.layers.Dense(128, activation="sigmoid")(layer)

	model_outputs = tf.keras.layers.Dense(1, activation="sigmoid")(layer)

	model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)
	model.compile(optimizer="adam",
	              loss="binary_crossentropy",
	              metrics=["binary_accuracy", "AUC", tf.keras.metrics.FalsePositives(), tf.keras.metrics.TruePositives()]
	              )
	return model


if __name__ == "__main__":
	start_time = time()
	EMB_DIM_COUNT = 256
	BATCH_SIZE = 128
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		"""
		graph retrieved from the page "https://snap.stanford.edu/data/egonets-Facebook.html", by downloading the file
		"facebook_combined.txt.gz". The full download link is "https://snap.stanford.edu/data/facebook_combined.txt.gz".
		"""
		read_graph_tstart = time()
		graph_file_name = "facebook_combined.txt"
		G, A = _read_graph(graph_file_name)
		print("Graph read in: {:.2f} seconds".format(time() - read_graph_tstart))
		emb_tstart = time()
		emb = get_nbne_embeddings(
									G,
									num_permutations=10,
									embedding_dimension=EMB_DIM_COUNT,
									window_size=5,
									min_count=0,
									min_degree=3,
									workers=8
									)
		# normalizing the embeddings should, generally, be a good a idea
		emb = normalize(emb)
		print("Vertices embeddings trained in: {:.2f} seconds".format(time() - emb_tstart))
		# here, one might use sampling_rate=1.0 to combine the links and non-links between all vertices, then the data is
		# split among training and testing by the sklearn.train_test_split
		assemble_tstart = time()
		x, y = assemble_balanced_link_prediction_data(A, emb, sampling_rate=0.10, dtype=np.float64)
		xtrain, xtest, ytrain, ytrue = train_test_split(x, y, test_size=1/4, shuffle=True)
		print("Training and testing data assembled in: {:.2f} seconds".format(time() - assemble_tstart))

		xgb_train_tstart = time()
		xgb_dict = xgboost_gpu.train_xgboost_gpu(
										xtrain,
										ytrain,
										data_chunksize=None,
										n_gpus=2,
										n_threads_per_gpu=10,
										params={"objective": "binary:logistic", "eval_metric": "logloss"}
										)
		xgb_model = xgb_dict['booster']
		xgb_train_telapsed = time() - xgb_train_tstart
		xgb_inference_tstart = time()

		ypredicted = xgboost_gpu.predict_xgboost_gpu(
										xgb_model,
										xtest,
										data_chunksize=None,
										n_gpus=1,
										n_threads_per_gpu=20,
										)
		ypredicted = np.round(ypredicted)
		xgb_inference_telapsed = time() - xgb_inference_tstart
		print_scores_summary(ytrue, ypredicted, xgb_train_telapsed, xgb_inference_telapsed, model_name="XGBoost (GPU)")

		xgb_inference_tstart = time()
		ypredicted = xgb_model.predict(xgboost.DMatrix(xtest, feature_names=[str(k) for k in range(xtest.shape[1])]))
		ypredicted = np.round(ypredicted)
		print("XGBoost (CPU) inference time: {:.2f} seconds.".format(time() - xgb_inference_tstart))

		# =============================================================================
		model = make_nn(EMB_DIM_COUNT * 2)
		print(model.summary())

		def scheduler(epoch, lr):
			return lr * tf.math.exp(-0.06)
		lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

		transformer_train_tstart = time()
		model.fit(xtrain, ytrain, epochs=50, batch_size=BATCH_SIZE, callbacks=[lr_scheduler])
		cnn_train_telapsed = time() - transformer_train_tstart

		cnn_inference_tstart = time()
		ypredicted = model.predict(xtest)
		ypredicted = np.round(ypredicted)
		cnn_iference_telapsed = time() - cnn_inference_tstart
		print_scores_summary(ytrue, ypredicted, cnn_train_telapsed, cnn_iference_telapsed, model_name="Transformer")

		print("Total script execution time: {:.2f} seconds.".format(time() - start_time))

