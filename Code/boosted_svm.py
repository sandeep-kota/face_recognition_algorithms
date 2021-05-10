import numpy as np 
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers
from utils import *
from svm import *
solvers.options['show_progress'] = False


def main():
	
	K_iters = 20
	classifiers = []
	classifier_errors = []
	classifier_weights = []
	train_acc_val = []
	test_acc_val = []

	X, y, X_test, y_test = loadExpData('data', cross_val_idx = 0, print_options=True)
	w = np.ones(X.shape[0])
	for i in range(K_iters):
		
		P_n = w / np.sum(w)
		
		if i==0:
			X_train = X
			y_train = y
		
		if i!=0:
			pick_idx = np.random.choice(np.arange(0,X.shape[0]),X.shape[0],p=P_n.ravel())
			X_train =  X[pick_idx, :]
			y_train = y[pick_idx]

		svm_obj = SVM(C_reg=1, thresh=1e-2, kernel='lin')
		svm_obj.fit(X_train,y_train)
		classifiers.append(svm_obj)
		
		svm_train_preds = svm_obj.predict(X_train).astype(np.int64)
		error = np.sum(P_n[svm_train_preds!=y_train])
		
		if error == 0:
			error = 1e-10

		classifier_errors.append(error)
		
		a = 0.5*np.log((1-error)/error)
		classifier_weights.append(a)
		w = (w/np.sum(w))*np.exp(-a*y_train*svm_train_preds)
		
		y_pred = 0
		for c in range(len(classifiers)):
			y_pred+=classifier_weights[c]*classifiers[c].predict(X_train)
		y_pred = np.sign(y_pred)
		train_acc = np.sum(y_pred==y_train)/len(y_train)
		train_acc_val.append(train_acc)

		y_pred = 0
		for c in range(len(classifiers)):
			y_pred+=classifier_weights[c]*classifiers[c].predict(X_test)
		y_pred = np.sign(y_pred)
		test_acc = np.sum(y_pred==y_test)/len(y_test)
		test_acc_val.append(test_acc)

		print("Iter :",i," Test Acc : ",test_acc, " Train Acc : ",train_acc )
		print("----------")
	
	## Visualize Training curve
	plt.plot(train_acc_val, color='r')
	plt.plot(test_acc_val, color= 'b')
	plt.xlabel('Number of iterations (K)');
	plt.ylabel('Accuracy');
	plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper left')
	plt.title('Boosted SVM Training Graph')
	plt.show()
	
if __name__=='__main__':
	main()