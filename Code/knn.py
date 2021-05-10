import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from utils import *
from MDA import *
from PCA import *

def KNN(X_train,k, x, print_options=False):
	squared_errors = []
	for i in range(X_train.shape[0]):
		for j in range(X_train.shape[1]):
			squared_errors.append([(i,np.mean((X_train[i,j]-x)**2))])
			# print(i,j,np.mean((X_train[i,j]-x)**2))
	squared_errors = np.array(squared_errors)
	squared_errors = np.reshape(squared_errors,(squared_errors.shape[0],squared_errors.shape[2]))
	squared_errors = squared_errors[squared_errors[:,1].argsort()]
	# squared_errors = np.sort(squared_errors, axis=0)
	knn_preds = squared_errors[:k,0]
	u, c = np.unique(knn_preds, return_counts=True)
	if print_options==True:
		print("KNN Prediction :", u[np.argmax(c)] )
	return u[np.argmax(c)]


def evalClassifier(X_train,k ,X_test,y_test,print_options=False):
	# print("Evaluating classifier....")
	y_pred = []
	for i in range(X_test.shape[0]): 
		y_pred.append(KNN(X_train, k=k, x=X_test[i]))
		if print_options==True:
			print("[y_true,y_pred] : [",y_test[i],",",int(y_pred[i]),"]")

	y_pred = np.array(y_pred)
	accuracy = accuracy_score(y_test,y_pred, normalize=True)
	if print_options==True:
		print("Accuracy :", accuracy)

	return accuracy



def main():
	# print_options = False
	print_options = True

	K = [1,3,5,7,9,11]
	table=[]
	for k in K: 
		print(k)
		avg_acc = 0
		for i in range(5):

			##<-------------Chose dataset----------->

			##Expression vs Neutral Face
			# X_train, y_train, X_test, y_test = loadExpMDA(n_test=50, print_options=True)

			## Face Recognition
			X_train, y_train, X_test, y_test = loadData('data', n_test=15, print_options=False)
			# X_train, y_train, X_test, y_test = loadData('pose', n_test=15, print_options=True)
			# X_train, y_train, X_test, y_test = loadData('illum', n_test=15, print_options=True)

			
			##<----------------------Dimensionality redunction--------------->

			#A_mtx = MDA(X_train=X_train, m_dims=10, lam_w_cov = 0.5, print_options = True)
			A_mtx = PCA(X_train=X_train, alpha=10, print_options = False)

			X_train = transformData(X_train, A_mtx)
			X_test = transformData(X_test, A_mtx)

			##<----------------KNN Classifier---------->

			accuracy = evalClassifier(X_train=X_train,k=k,X_test=X_test,y_test=y_test,print_options=False)
			# print(accuracy)
			avg_acc+=accuracy
		table.append(avg_acc/5)

	print(table)

if __name__=='__main__':
	main()

