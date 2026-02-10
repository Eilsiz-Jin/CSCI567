import numpy as np
import json
import collections

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def data_processing(data):
	train_set, valid_set, test_set = data['train_data'], data['val_data'], data['test_data']
	Xtrain = train_set["features"]
	ytrain = train_set["labels"]
	Xval = valid_set["features"]
	yval = valid_set["labels"]
	Xtest = test_set["features"]
	ytest = test_set["labels"]

	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)

	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	return Xtrain, ytrain, Xval, yval, Xtest, ytest


def data_processing_with_transformation(data, do_minmax_scaling=True, do_normalization=False):
	train_set, valid_set, test_set = data['train_data'], data['val_data'], data['test_data']
	Xtrain = train_set["features"]
	ytrain = train_set["labels"]
	Xval = valid_set["features"]
	yval = valid_set["labels"]
	Xtest = test_set["features"]
	ytest = test_set["labels"]

	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)

	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	# We load data from json here and turn the data into numpy array
	# You can further perform data transformation on Xtrain, Xval, Xtest

	# Min-Max scaling
	if do_minmax_scaling:
		#####################################################
		#				 YOUR CODE HERE					    #
		
		min = Xtrain.min(axis=0)
		max = Xtrain.max(axis=0)
		Xtrain = (Xtrain - min) / (max - min + 1e-6)
		Xval = (Xval - min) / (max - min + 1e-6)
		Xtest = (Xtest - min) / (max - min + 1e-6)
		#####################################################

	# Normalization
	def normalization(x):
		#####################################################
		#				 YOUR CODE HERE					    #
		return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)
		#####################################################
	
	if do_normalization:
		Xtrain = normalization(Xtrain)
		Xval = normalization(Xval)
		Xtest = normalization(Xtest)

	return Xtrain, ytrain, Xval, yval, Xtest, ytest


def compute_l2_distances(Xtrain, X):
	"""
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
	"""
	#####################################################
	#				 YOUR CODE HERE					    #

	num_train = Xtrain.shape[0]
	num_test = X.shape[0]
	dists = np.zeros((num_test, num_train))
	for i in range(num_test):
		for j in range(num_train):
			dists[i][j] = np.sqrt(np.sum((X[i] - Xtrain[j]) ** 2))
	#####################################################
	return dists


def compute_cosine_distances(Xtrain, X):
	"""
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Cosine distance between the ith test point and the jth training
	  point.
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	num_test = X.shape[0]
	num_train = Xtrain.shape[0]
	dists = np.zeros((num_test, num_train))
	# breakpoint()
	for i in range(num_test):
		for j in range(num_train):
			if np.linalg.norm(X[i]) == 0 or np.linalg.norm(Xtrain[j]) == 0:
				dists[i][j] = 1
			else:
				dists[i][j] = 1 - np.dot(X[i], Xtrain[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(Xtrain[j]) + 1e-6)
	#####################################################
	return dists


def predict_labels(k, ytrain, dists):
	"""
	Given a matrix of distances between test points and training points,
	predict a label for each test point.
	Inputs:
	- k: The number of nearest neighbors used for prediction.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  gives the distance betwen the ith test point and the jth training point.
	Returns:
	- ypred: A numpy array of shape (num_test,) containing predicted labels for the
	  test data, where y[i] is the predicted label for the test point X[i].
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	num_test = dists.shape[0]
	ypred = np.zeros(num_test, dtype=int)
	sorted_indices = np.argsort(dists, axis=1)
	first_k_indices = sorted_indices[:, :k]

	for i in range(num_test):
		ytrain_k = ytrain[first_k_indices[i]]
		ypred[i] = np.bincount(ytrain_k).argmax()
	#####################################################
	return ypred


def compute_error_rate(y, ypred):
	"""
	Compute the error rate of prediction based on the true labels.
	Inputs:
	- y: A numpy array with of shape (num_test,) where y[i] is the true label
	  of the ith test point.
	- ypred: A numpy array with of shape (num_test,) where ypred[i] is the
	  prediction of the ith test point.
	Returns:
	- err: The error rate of prediction (scalar).
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	total = len(y)
	error_num = 0
	for i in range(total):
		if y[i] != ypred[i]:
			error_num += 1
	err = (error_num / total)
	#####################################################
	return err


def find_best_k(K, ytrain, dists, yval):
	"""
	Find best k according to validation error rate.
	Inputs:
	- K: A list of ks.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the distance between the ith test point and the jth training
	  point.
	- yval: A numpy array with of shape (num_val,) where y[i] is the true label
	  of the ith validation point.
	Returns:
	- best_k: The k with the lowest error rate.
	- validation_error: A list of error rate of different ks in K.
	- best_err: The lowest error rate we get from all ks in K.
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	best_err = 1.0
	validation_error = []
	best_k = K[0]
	for k in K:
		ypred = predict_labels(k, ytrain, dists)
		err = compute_error_rate(yval, ypred)
		validation_error.append(err)
		if err < best_err:
			best_err = err
			best_k = k
	#####################################################
	return best_k, validation_error, best_err


def main():
	input_file = 'breast_cancer_dataset.json'
	output_file = 'knn_output.txt'

	#==================Problem Set 1.1=======================

	with open(input_file) as json_data:
		data = json.load(json_data)

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.1")
	print()

	#comment#: 
	# The validation error rate is 0.07692307692307693 (7.69%) when k = 4 using Euclidean distance.

	#==================Problem Set 1.2=======================

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_with_transformation(data, do_minmax_scaling=False, do_normalization=True)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.2 when using normalization")
	print()

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_with_transformation(data, do_minmax_scaling=True, do_normalization=False)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.2 when using minmax_scaling")
	print()
	
	#comment#: 
	# The validation error rate is 0.04395604395604396 (4.40%) in Problem Set 1.2 when using normalization
	# The validation error rate is 0.05494505494505494 (5.49%) in Problem Set 1.2 when using minmax_scaling
	#==================Problem Set 1.3=======================

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)
	dists = compute_cosine_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.3, which use cosine distance")
	print()

	#comment#: 
	# The validation error rate is 0.04395604395604396 (4.40%) in Problem Set 1.3, which use cosine distance
	#==================Problem Set 1.4=======================
	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)

	#======performance of different k in training set=====
	K = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]
	#####################################################
	#				 YOUR CODE HERE					    #

	dists = compute_l2_distances(Xtrain, Xtrain)
	train_err = []
	for k in K:
		ypred_train = predict_labels(k, ytrain, dists)
		err_train = compute_error_rate(ytrain, ypred_train)
		train_err.append(err_train)

	plt.plot(K, train_err, marker='o', label='Training Error Rate')
	plt.xlabel('k')
	plt.ylabel('Error Rate')
	plt.title('Training Error Rate vs k')
	plt.savefig('Train_Error_vs_k.png')
	plt.close()

	dists = compute_l2_distances(Xtrain, Xval)
	train_err = []
	for k in K:
		ypred_train = predict_labels(k, ytrain, dists)
		err_train = compute_error_rate(yval, ypred_train)
		train_err.append(err_train)

	plt.plot(K, train_err, marker='o', label='Validation Error Rate')
	plt.xlabel('k')
	plt.ylabel('Error Rate')
	plt.savefig('Val_Error_vs_k.png')
	plt.close()
	#####################################################

	#==========select the best k by using validation set==============
	dists = compute_l2_distances(Xtrain, Xval)
	best_k, validation_error, best_err = find_best_k(K, ytrain, dists, yval)

	#===============test the performance with your best k=============
	dists = compute_l2_distances(Xtrain, Xtest)
	ypred = predict_labels(best_k, ytrain, dists)
	test_err = compute_error_rate(ytest, ypred)
	print("In Problem Set 1.4, we use the best k = ", best_k, "with the best validation error rate", best_err)
	print("Using the best k, the final test error rate is", test_err)
	#====================write your results to file===================
	f=open(output_file, 'w')
	for i in range(len(K)):
		f.write('%d %.3f' % (K[i], validation_error[i])+'\n')
	f.write('%s %.3f' % ('test', test_err))
	f.close()

	#comment#:
	# 1) The error rate generally shows an upward trend as k increases on the training set.
	# 2) The Best k is 6 on validation set.
	# 3) On the training set, erorr rate becomes worse when k is bigger. But on validation set, the error rate shows a trend of first decreasing and then increasing as k increases. 
	# 4) The final test set error rate is 0.07100591715976332 (7.10%) when k is 6 (best-k).
	# 5) As k increases, the training error monotonically increases because the KNN classifier becomes less flexible and the decision boundary becomes smoother, leading to underfitting.
	# The validation error first decreases and then increases as k grows. For small k, the model overfits the training data and generalizes poorly to unseen data. For large k, the model becomes too simple and fails to capture the underlying structure of the data, resulting in underfitting.
	# The validation set is used for hyper-parameter tuning because it provides an unbiased estimate of generalization performance, while the training set error is optimistically biased.
	# By selecting k based on the minimum validation error, we achieve a good balance between overfitting and underfitting, leading to better generalization performance on the test set.

if __name__ == "__main__":
	main()
