import numpy as np
import json
import collections
import matplotlib.pyplot as plt


np.random.seed(42) ## random seed fixed 

d = 100 # dimensions of data
n = 1000 # number of data points
X = np.random.normal(0,1, size=(n,d))
X_test = np.random.normal(0,1,size=(n,d)) 
w_true = np.random.normal(0,1, size=(d,1))
y = X.dot(w_true) + np.random.normal(0,0.5,size=(n,1))
y_test = X_test.dot(w_true) + np.random.normal(0,0.5,size=(n,1))

#########   Do not change the code above  ############
######################################################

def square_loss(w,X,y):
	"""
	Implement total squared error given weight w, dataset (X,y)
	Inputs:
	- w: weight for the linear function
	- X: dataset of size (n,d)
	- y: label of size (n,1) 
	Returns:
	- loss: total squared error of w on dataset (X,y)
	"""
	################################
	##     Write your code here   ##
	################################


#### Implement closed-form solution given dataset (X,y)
def closed_form(X,y):
	"""
	Implement closed-form solution given dataset (X,y)
	Inputs:
	- X: dataset of size (n,d)
	- y: label of size (n,1) 
	Returns:
	- w_LS: closed form solution of the weight
	- loss: total squared error of w_LS on dataset (X,y)
	"""
	################################
	##     Write your code here   ##
	################################


def gradient_descent(X, y, lr_set, N_iteration):
    """
    Implement gradient descent on the square-error given dataset (X, y) for each learning rate in lr_set.
    Inputs:
    - X: dataset of size (n, d)
    - y: label of size (n, 1)
    - lr_set: a list of learning rates
    - N_iteration: the number of iterations
    Returns:
    - a plot with k curves where k is the length of lr_set
    - each curve contains 20 data points, in which the i-th data point represents the total squared-error
      with respect to the i-th iteration
    - You can print the final objective value within this function to show the performance of the best step size
    """
   	################################
	##     Write your code here   ##
	################################


def stochastic_gradient_descent(X,y,lr_set,N_iteration):
	"""
	Implement gradient descent on the square-error given dataset (X,y) and for each learning rate in lr_set
	Inputs:
	- X: dataset of size (n,d)
	- y: label of size (n,1)
	- lr_set: a list of learning rate.
	- N_itertion: the number of iterations
	Returns:
	- a plot with k curves where k is the length of lr_set.
	- each curve contains 1000 data points, in which the i-th data point represents the total squared-error with respect to the i-th iteration
	- You can print the final objective value within this function to show the performance of best step size
	"""
	np.random.seed(1) # Use this fixed random_seed in sampling
	################################
	##     Write your code here   ##
	################################


def main():
	### Problem 4.1 ###
	w_LS, loss_LS_train = closed_form(X,y)
	w_0 = np.zeros((d,1))
	loss_0_train = square_loss(w_0,X,y)
	loss_LS_test = square_loss(w_LS,X_test, y_test)

	print("F(w_LS)=", loss_LS_train, " on training data")
	print("F(w_0)=", loss_0_train, " on training data")
	print("F(w_LS)=", loss_LS_test, " on testing data")


	### Problem 4.2 (Gradient Descent) ###
	### You can plot more options of lr_set if necessary
	lr_set = [0.00005, 0.0005, 0.0007]
	w_0 = np.zeros((n,1))
	N_iter = 20
	gradient_descent(X,y,lr_set,N_iter)

	### Problem 4.3 (Stochastic Gradient Descent) ###
	### You can plot more options of lr_set if necessary
	lr_set = [0.0005, 0.005, 0.01]
	w_0 = np.zeros((n,1))
	N_iter = 1000
	stochastic_gradient_descent(X,y,lr_set,N_iter)


if __name__ == "__main__":
	main()