import numpy as np
import pandas as pd
import networkx as nx
import numba as nb
from nbne import train_model
from collections import defaultdict
from os.path import join, abspath, dirname
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from time import time
import xgboost


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


@nb.njit(fastmath=True)
def build_link_prediction_train(A, emb, sampling_rate=0.5, dtype=np.float64):
	'''
	Generates training/testing data for graph "A" using it's embeddings "emb". This function is meant to be used with
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
	num_nodes = A.shape[0]
	emb_dimms = emb.shape[1]
	expected_instances = int(sampling_rate * (((num_nodes ** 2) - num_nodes) / 2))
	if expected_instances <= 0:
		expected_instances = 1
	elif expected_instances > ((num_nodes ** 2) - num_nodes):
		expected_instances = (num_nodes ** 2) - num_nodes

	selected_nodes = np.empty(shape=(int(((num_nodes ** 2) - num_nodes) / 2), 2), dtype=np.int64)
	k = 0
	for i in range(num_nodes - 1):
		for j in range(i + 1, num_nodes):
			selected_nodes[k, 0] = i
			selected_nodes[k, 1] = j
			k += 1
	np.random.shuffle(selected_nodes)
	selected_nodes = selected_nodes[:expected_instances]

	x = np.empty(shape=(expected_instances, 2 * emb_dimms), dtype=dtype)
	y = np.empty(shape=expected_instances, dtype=dtype)
	for k in range(expected_instances):
		i = selected_nodes[k, 0]
		x[k, :emb_dimms] = emb[i]
		j = selected_nodes[k, 1]
		x[k, emb_dimms:] = emb[j]
		y[k] = A[i, j]
	return x, y


if __name__ == "__main__":
	start_time = time()
	"""
	This invoking of the "build_link_prediction_train(...)" is meant only to force numba to compile the function to C
	without including the meaningful compiling time in the final printed time. This method of compiling ahead of time
	is not recommended, it's just a quick and dirty version, for the right way to do so, check numba's documentation:
	"https://numba.pydata.org/numba-doc/dev/user/pycc.html".
	"""
	build_link_prediction_train(np.empty(shape=(2, 2)), np.empty(shape=(2, 2)))
	print("Numba's compilation time of the \"build_link_prediction_train(...)\" function: {:.2f} seconds"
				.format(time() - start_time))
	start_time = time()
	"""
	graph retrieved from the page "https://snap.stanford.edu/data/egonets-Facebook.html", by downloading the file
	"facebook_combined.txt.gz". The full download link is "https://snap.stanford.edu/data/facebook_combined.txt.gz".
	"""
	graph_file_name = "facebook_combined.txt"
	G, A = _read_graph(graph_file_name)
	emb = get_nbne_embeddings(
								G,
								num_permutations=10,
								embedding_dimension=8,
								window_size=5,
								min_count=0,
								min_degree=0,
								workers=8
								)
	# normalizing the embeddings should, generally, be a good a idea
	emb = normalize(emb)
	# here, one might use sampling_rate=1.0 to combine the links and non-links between all vertices, then the data is
	# split among training and testing by the sklearn.train_test_split
	x, y = build_link_prediction_train(A, emb, sampling_rate=0.10, dtype=np.float64)
	xtrain, xtest, ytrain, ytrue = train_test_split(x, y, test_size=1/2, shuffle=True)
	xgb_model = xgboost.XGBClassifier()
	xgb_model.fit(xtrain, ytrain)
	ypredicted = xgb_model.predict(xtest)

	print("Accuracy: {:.2f}%".format(accuracy_score(ytrue, ypredicted) * 100))
	print("ROC AUC: {:.2f}%".format(roc_auc_score(ytrue, ypredicted) * 100))
	print("Total elapsed time without considering numba's compilation: {:.2f} seconds.".format(time() - start_time))

