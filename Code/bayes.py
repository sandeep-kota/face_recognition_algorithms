import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.metrics import accuracy_score
from utils import *
from MDA import *
from PCA import *

def bayesClassifier(X_train, lam, print_options=False):

	mean_ML = np.mean(X_train,axis=1)

	if print_options==True:
		print("X_train shape :", X_train.shape)
		print("mean_ML shape :", mean_ML.shape)

		# # Plot sample data
		# fig,a =  plt.subplots(5,8)
		# for j in range(0,5):
		# 	for i in range(7):
		# 		a[j,i].imshow(np.reshape(X_train[j,i],(48,40)), cmap='gray')
		# 		a[j,i].axis('off')
		# 		a[j,i].set_title(y_train[j])
		# 	a[j,7].imshow(np.reshape(mean_ML[j],(48,40)), cmap='gray')
		# 	a[j,7].axis('off')
		# 	a[j,7].set_title(y_train[j])
		# plt.show()

	sigma_ML = []
	
	for i in range(X_train.shape[0]):
		sigma_ML.append(np.cov(X_train[i].T) + (lam*np.eye(X_train.shape[-1])))
	sigma_ML=np.array(sigma_ML)

	# for i in range(sigma_ML.shape[0]):
	# 	print(i,np.linalg.det(sigma_ML[i]))
	# sys.exit(0)
	
	if print_options==True:
		print("sigma_ML shape :", sigma_ML.shape)

	return mean_ML,sigma_ML


def evalClassifier(X_train,mean_ML,sigma_ML,X_test,y_test,print_options=False):
	
	print("Evaluating classifier....", X_test.shape)
	y_pred = []
	for x in range(X_test.shape[0]): 
		posterior = []
		for i in range(X_train.shape[0]):
			prior = 1/(X_train.shape[1])
			posterior.append(prior*mvn.logpdf(X_test[x],mean=mean_ML[i], cov=sigma_ML[i]))
		posterior = np.array(posmvn.logpdf(X_test[x],mean=mean_ML[i], cov=sigma_ML[i])terior)
		if print_options==True:
			print("Idx : ",x+1, " True Label, Prediction :", y_test[x],np.argmax(posterior))
		y_pred.append(np.argmax(posterior))


	y_pred = np.array(y_pred)
	accuracy = accuracy_score(y_test,y_pred, normalize=True)
	if print_options==True:
		print("Accuracy :", accuracy)

	return accuracy

def main():
	
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

	##<----------------Bayes Classifier---------->
	# mean_ML, sigma_ML = bayesClassifier(X_train,lam = 0.4, print_options=True)
	mean_ML, sigma_ML = bayesClassifier(X_train,lam = 0.96, print_options=True)
	accuracy = evalClassifier(X_train,mean_ML,sigma_ML,X_test,y_test,print_options=True)

	
if __name__=='__main__':
	main()

