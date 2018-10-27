import numpy as np

###
#	Binary Classification
#	The most primitive version	
#	Step Function is used
#	Labels are : 1 and -1
# 	Update Rule : W(t+1) <- W(t) + (True Label) * (Predicted Label)
###
class Perceptron:
	
	def __init__(self, max_iter = 300, verbose = False):
		self.max_iter = max_iter
		self.verbose  = verbose

		self.loss = 0
		self.weights = None

		self.vectorized_step_func = np.vectorize(lambda x: -1 if x < 0 else 1)
		self.vectorized_equal_func = np.vectorize(lambda x, y: 1 if x != y else 0)

	###
	#	data = [ [--X1--],	labels = [Y1, Y2, ... , Ym]
	#			 [--X2--],
	#			 ...	 ,
	#			 [--Xm--] ]
	###
	def fit(self, data, labels):
		self.X = np.insert(arr = data, obj = 0, values = 1, axis = 1)
		self.Y = labels
		self.weights = np.zeros([1, data.shape[1]+1]) 
		loss_history = np.zeros(self.max_iter)
		print("\n")
		for iter in range(self.max_iter):
			total = self.X.dot(self.weights.T)
			hypothesis = self.vectorized_step_func(total.T)
			label_comparison = self.vectorized_equal_func(hypothesis, self.Y)
			update = (self.Y * label_comparison).dot(self.X)
			self.weights += update 
			self.loss = label_comparison.sum()
			loss_history[iter] = self.loss
			print("Iter {} Loss : {}".format(iter, self.loss))
			if self.loss == 0:
				break
			if self.verbose == True:
				print("Weights : \n{}".format(self.weights))
				print("Predictions : {}".format(hypothesis))
				print("Labels : {}".format(label_comparison))
				print("Update : \n{}".format(update))


	def predict(self, data):
		total = np.insert(arr = data, obj = 0, values = 1).dot(self.weights.T)
		return -1 if total < 0 else 1

	###	
	# 	Returns the mean accuracy given the training set in data and labels
	###
	def score(self, data, labels):
		total = np.insert(arr = data, obj = 0, values = 1, axis = 1).dot(self.weights.T)
		hypothesis = self.vectorized_step_func(total.T)
		label_comparison = self.vectorized_equal_func(hypothesis, labels)
		return 1 - label_comparison.sum() / label_comparison.shape[1]


import random
import matplotlib.pyplot as plt
import sklearn.linear_model 

if __name__ == "__main__":
	p = Perceptron(verbose = False)
	m = 700
	X = []
	Y = []
	for i in range(0,m):
		X.append([random.randint(-500,100),random.randint(-500,100)])
		Y.append(-1)
		X.append([random.randint(-100,500),random.randint(-100,500)])
		Y.append(1)

	X = np.array(X)
	Y = np.array(Y)

	#plt.scatter(X[:,0], X[:,1])
	#plt.show()
	p.fit(X, Y)

	mtest = 300
	testX = []
	testY = []
	for i in range(0,mtest):
		testX.append([random.randint(-500,100),random.randint(-500,100)])
		testY.append(-1)
		testX.append([random.randint(-100,500),random.randint(-100,500)])
		testY.append(1)

	testX = np.array(testX)
	testY = np.array(testY)

	print("Score : ", p.score(testX, testY))

	p2 = sklearn.linear_model.Perceptron(max_iter = p.max_iter, tol=1e-3)
	p2.fit(X,Y.T)
	print("Score Scikit : ", p2.score(testX, testY.T))




