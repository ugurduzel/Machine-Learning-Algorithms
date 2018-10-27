import numpy as np
import matplotlib.pyplot as plt

###
#	Ordinary least squares Linear Regression
#	Scikit does not implement bias term (w0 = 1)
#	It can be added easily by self.X = np.insert(arr = data, obj = 0, values = 1)
#	And to line 52 as return np.insert(arr = data, obj = 0, values = 1).dot(self.weights.T)
###
class LinearRegression:

	def __init__(self, learning_rate = 0.001, max_epochs = 300, verbose = False):
		self.verbose  = verbose
		self.max_epochs = max_epochs
		self.lr = learning_rate

		self.loss = 0
		self.weights = None

	def gradientDescent(self):
	    cost_history = np.zeros(self.max_epochs)
	    for iter in range(self.max_epochs):
	        hypothesis = self.X.dot(self.weights.T)
	        update = ( (self.lr/self.m) * self.X.T.dot((hypothesis - self.Y))).reshape(1,self.X.shape[1])
	        self.weights = self.weights - update
	        self.loss = np.sum(np.square(hypothesis - self.Y)) / (2*self.m)
	        if self.verbose == True:
	        	print("Iter {} Loss : {}".format(iter, self.loss))
	        cost_history[iter] = self.loss 
	    return cost_history

	###
	#	data = [ [--X1--],	labels = [Y1, Y2, ... , Ym]
	#			 [--X2--],
	#			 ...	 ,
	#			 [--Xm--] ]
	###
	def fit(self, data, labels):
		self.X = np.array(data)
		self.m = self.X.shape[0]
		self.Y = labels.reshape(self.m, 1)
		self.weights = np.zeros([1,data.shape[1]]) 
		costs = self.gradientDescent()
		if self.verbose == True:
			plt.plot(np.arange(self.max_epochs), costs)
			plt.show()
	
	
	def predict(self, data):
		return np.array(data).dot(self.weights.T)





import random
import sklearn.preprocessing
import sklearn.linear_model 
import sklearn.metrics

if __name__ == "__main__":
	model = LinearRegression(learning_rate = 0.0001, max_epochs = 300, verbose = True)
	m = 10000
	X = []
	Y = []
	for i in range(0,m):
		X.append([random.randint(10,20)])
		Y.append(random.randint(10,20))
	
	X = np.array(X)
	Y = np.array(Y)

	#plt.scatter(X,Y)
	#plt.show()
	model.fit(X, Y)

	mtest = 3000
	testX = []
	testY = []
	for i in range(0,mtest):
		testX.append([random.randint(30,60)])
		testY.append(random.randint(30,60))


	testX = np.array(testX)
	testY = np.array(testY)

	print("Score : ", sklearn.metrics.r2_score(testY.T, model.predict(testX)))

	model2 = sklearn.linear_model.LinearRegression(fit_intercept=False)
	model2.fit(X,Y.T)
	print("Score Scikit : ", sklearn.metrics.r2_score(testY.T, model2.predict(testX)))



