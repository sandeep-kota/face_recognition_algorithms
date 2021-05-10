import numpy as np 
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers
from utils import *

## Define kernel RBF
def rbfKernel(x,y,k_params=0.5):
	x = np.expand_dims(x,axis=1)
	y = np.expand_dims(y,axis=1)
	val = np.exp(-(np.linalg.norm(x-y)**2)/k_params**2)
	val = np.array(val)
	val = np.reshape(val,(1,1))
	return val

def polyKernel(x,y,k_params=2):
	x = np.expand_dims(x,axis=1)
	y = np.expand_dims(y,axis=1)
	val = (np.matmul(x.T, y) + 1) ** k_params
	return val

def linKernel(x,y):
	x = np.expand_dims(x,axis=1)
	y = np.expand_dims(y,axis=1)
	val = (np.matmul(x.T, y))
	return val


class SVM(object):
	def __init__(self, C_reg=0.5, thresh=1e-5, kernel='poly', k_params=3):
		super(SVM,self).__init__()
		self.C_reg = C_reg
		self.thresh = thresh
		self.alpha = None
		self.intercept = None
		self.sv = None
		self.sv_l = None
		self.kernel = kernel
		self.k_params = k_params

	def fit(self, X_train,y_train, print_options=False):
		P = np.zeros((X_train.shape[0],X_train.shape[0]))

		for i in range(X_train.shape[0]):
			for j in range(X_train.shape[0]):
				if self.kernel=='rbf':
					P[i,j] = 1*y_train[i]*y_train[j]*rbfKernel(X_train[i],X_train[j],k_params=self.k_params)
				if self.kernel=='poly':
					P[i,j] = 1*y_train[i]*y_train[j]*polyKernel(X_train[i],X_train[j],k_params=self.k_params)
				if self.kernel=='lin':
					P[i,j] = 1*y_train[i]*y_train[j]*linKernel(X_train[i],X_train[j])
		q =  -1*np.ones((X_train.shape[0],1))
		G = np.vstack((-1*np.eye(X_train.shape[0]),np.eye(X_train.shape[0])))
		h = np.vstack((np.zeros((X_train.shape[0],1)),self.C_reg*np.ones((X_train.shape[0],1))))
		A = np.expand_dims(np.copy(y_train),axis=0)
		b = np.zeros((1,1))

		if print_options==True:
			print("P :", P.shape)
			print("q :", q.shape)
			print("G :", G.shape)
			print("h :", h.shape)
			print("A :", A.shape)
			print("b :", b.shape)


		P = matrix(P,tc='d')
		q = matrix(q,tc='d')
		G = matrix(G,tc='d')
		h = matrix(h,tc='d')
		A = matrix(A,tc='d')
		b = matrix(b,tc='d')

		solvers.options['show_progress'] = False

		solution = solvers.qp(P=P,q=q,G=G,h=h,A=A,b=b)
		self.alpha = np.array(solution['x'])
		if print_options==True:
			print("Solutions :", self.alpha[:5])
		
		idx = np.arange(X_train.shape[0])[np.ravel(self.alpha>self.thresh)]
		self.alpha = self.alpha[idx]

		self.sv = X_train[idx]
		self.sv_l = y_train[idx]

		if print_options==True:
			print("Number of Support Vectors :",self.sv.shape[0])

		K_mtx = []
		for i in range(self.sv.shape[0]):
			if self.kernel == 'rbf':
				K_mtx.append(rbfKernel(self.sv[i],X_train[0],k_params=self.k_params))
			if self.kernel == 'poly':
				K_mtx.append(polyKernel(self.sv[i],X_train[0],k_params=self.k_params))
			if self.kernel=='lin':
				K_mtx.append(linKernel(self.sv[i],X_train[0]))
		K_mtx = np.array(K_mtx)
		K_mtx = np.squeeze(K_mtx,axis=-1)

		self.intercept=self.sv_l[0]-np.sum(np.ravel(self.alpha)*np.ravel(self.sv_l)*np.ravel(K_mtx))

		if print_options==True:
			print("alpha shape, value :", self.alpha.shape,self.alpha[0])
			print("intercept:", self.intercept)

	def eval(self,X_test, y_test):
		y_pred=[]
		for x in range(X_test.shape[0]):
			K_mtx = []
			for i in range(self.sv.shape[0]):
				if self.kernel == 'rbf':
					K_mtx.append(rbfKernel(self.sv[i],X_test[x],k_params=self.k_params))
				if self.kernel == 'poly':
					K_mtx.append(polyKernel(self.sv[i],X_test[x],k_params=self.k_params))
				if self.kernel=='lin':
					K_mtx.append(linKernel(self.sv[i],X_test[x]))
			K_mtx = np.array(K_mtx)
			y_pred.append(np.sign(np.sum(np.ravel(self.alpha)*np.ravel(self.sv_l)*np.ravel(K_mtx))+self.intercept))
		y_pred = np.array(y_pred)
		acc = np.sum(np.ravel(y_test) == np.ravel(y_pred)) / len(y_test) 
		return acc

	def predict(self,X_test):
		y_pred=[]
		for x in range(X_test.shape[0]):
			K_mtx = []
			for i in range(self.sv.shape[0]):
				if self.kernel == 'rbf':
					K_mtx.append(rbfKernel(self.sv[i],X_test[x],k_params=self.k_params))
				if self.kernel == 'poly':
					K_mtx.append(polyKernel(self.sv[i],X_test[x],k_params=self.k_params))
				if self.kernel=='lin':
					K_mtx.append(linKernel(self.sv[i],X_test[x]))
			K_mtx = np.array(K_mtx)
			y_pred.append(np.sign(np.sum(np.ravel(self.alpha)*np.ravel(self.sv_l)*np.ravel(K_mtx))+self.intercept))
		y_pred = np.array(y_pred)
		return y_pred

def main():

	# k_params = [1,2,3,4,5]	# parameters for Polynomial Kernel
	k_params = [2,4,6,8,10] #Parameters for RBF Kernel
	# k_params = [1] # Dummy parameter for linear kernel
	
	for k_val in k_params:

		## Use the function call corresponding to the kernle
		
		# svm_obj = SVM(C_reg=1, thresh=10**(-k_val*3), kernel='poly', k_params=k_val)
		svm_obj = SVM(C_reg=1, thresh=1e-2, kernel='rbf', k_params=k_val)
		# svm_obj = SVM(C_reg=10, thresh=1e-5, kernel='lin')
		
		## SVM with 4 fold cross validation
		mean_acc = 0
		for cv_idx in range(4):
			X_train, y_train, X_test, y_test = loadExpData('data', cross_val_idx=cv_idx ,print_options=False)
			svm_obj.fit(X_train=X_train,y_train=y_train, print_options=False)
			acc = svm_obj.eval(X_test=X_test, y_test=y_test)
			mean_acc+=acc
			pred = svm_obj.predict(X_test)
			print("Accuracy ", cv_idx , ":",acc)
		print("Avg accuracy :", mean_acc/4," Param :", k_val)
		print("-----------")

if __name__=='__main__':
	main()