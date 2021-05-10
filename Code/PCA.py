import numpy as np 
import matplotlib.pyplot as plt
import sys
from utils import *

def PCA(X_train, alpha, print_options=False):
	#Step 1: Center the data
	mean = np.mean(np.mean(X_train,axis=1),axis=0)
	X_train = X_train-mean
	X_train = np.reshape(X_train,(X_train.shape[0]*X_train.shape[1],X_train.shape[2]))
	if print_options==True:
		print("X_train shape :",X_train.shape)

	#Visualize X_norm
	# fig,a =  plt.subplots(4,3)
	# for i in range(4):
	# 	for j in range(2):
	# 		a[i,j].imshow(np.reshape(X_train[i,j],(24,21)), cmap='gray')
	# 		a[i,j].set_title(y_train[i])
	# 		a[i,j].axis('off')
	# 	a[i,2].imshow(np.reshape(mean,(24,21)),cmap='gray')
	# 	a[i,2].axis('off')
	# plt.show()
	
	U,S,VT = np.linalg.svd(X_train.T,full_matrices=False)
	if print_options==True:

		print("U shape :", U.shape)
		print("S shape :", S.shape)
		print("VT shape :", VT.shape)

	eig_val_sum = 0
	for i in range(S.shape[0]):
		eig_val_sum+=S[i]
		if (eig_val_sum/np.sum(S)) >= (1-(alpha/100)):
			m = i
			break

	A = U[:,:m].T

	if print_options==True:
		print("A shape :",A.shape)

	## Visualize X_norm
	# fig,a =  plt.subplots(2,4)
	# for j in range(4):
	# 	for i in range(2):
	# 		a[i,j].imshow(np.reshape(A[4*i+j],(24,21)), cmap='gray')
	# 		# a[i,j].imshow(np.reshape(A[4*i+j],(48,40)), cmap='gray')
	# 		a[i,j].set_title('Eigenface ' + str(4*i+j))
	# 		a[i,j].axis('off')
	# plt.show()

	return A.T

def main():
	print_options = True
	
	## Chose dataset

	##Expression vs Neutral Face
	X_train, y_train, X_test, y_test = loadExpMDA(n_test=30, print_options=True)

	## Face Recognition
	# X_train, y_train, X_test, y_test = loadData('data', n_test=5, print_options=False)
	# X_train, y_train, X_test, y_test = loadData('pose', n_test=5, print_options=True)
	# X_train, y_train, X_test, y_test = loadData('illum', n_test=5, print_options=True)

	A_mtx = PCA(X_train=X_train, alpha=5, print_options=True)

	t_X_train = transformData(X_train, A_mtx)
	t_X_test = transformData(X_test, A_mtx)

	if print_options==True:
		print("Transformed data :",t_X_train.shape)
		print("Transformed data :",t_X_test.shape)


if __name__=='__main__':
	main()