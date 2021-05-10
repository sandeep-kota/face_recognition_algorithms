import numpy as np 
import matplotlib.pyplot as plt
import sys
from utils import *


def MDA(X_train, m_dims, lam_w_cov=0.2 , print_options = False):
	
	## Step 1: Compute emperical means of each class
	e_mean = np.mean(X_train,axis=1)
	tot_mean = np.mean(e_mean,axis=0)
	prior = 1/X_train.shape[0]# Constant
	if print_options==True:
		print("e_mean shape:", e_mean.shape)
		print("tot_mean shape:", tot_mean.shape)
		print("prior:", prior)
	
	# plt.imshow(np.reshape(tot_mean,(24,21)),cmap='gray')
	# plt.show()
	w_cov = np.zeros((X_train.shape[-1],X_train.shape[-1]))
	b_cov = np.zeros((X_train.shape[-1],X_train.shape[-1]))
	for i in range(X_train.shape[0]):
		X_i = X_train[i]
		i_mean = np.mean(X_i,axis=0)
		w_cov += prior*np.matmul((X_i-i_mean).T,(X_i-i_mean))*(1/X_train.shape[1]) + lam_w_cov*np.eye(X_train.shape[-1])		
		b_cov += prior*np.outer((i_mean-tot_mean),(i_mean-tot_mean))
	
	if print_options==True:
		print("b_cov shape:", b_cov.shape)
		print("w_cov shape:", w_cov.shape)
	
	sq_mtx = np.matmul(np.linalg.inv(w_cov ),b_cov)
	U,S,VT = np.linalg.svd(sq_mtx,full_matrices=False)
	if print_options==True:

		print("U shape :", U.shape)
		print("S shape :", S.shape)
		print("VT shape :", VT.shape)
	A_mtx = U[:,:m_dims]
	
	# fig,a =  plt.subplots(2,4)
	# for j in range(4):
	# 	for i in range(2):
	# 		a[i,j].imshow(np.reshape(A_mtx.T[4*i+j],(24,21)), cmap='gray')
	# 		# a[i,j].imshow(np.reshape(A_mtx.T[4*i+j],(48,40)), cmap='gray')
	# 		a[i,j].set_title('MDA  ' + str(4*i+j))
	# 		a[i,j].axis('off')
	# plt.show()

	return A_mtx

def main():
	print_options = True

	## Chose dataset

	##Expression vs Neutral Face
	X_train, y_train, X_test, y_test = loadExpMDA(n_test=30, print_options=True)

	## Face Recognition
	# X_train, y_train, X_test, y_test = loadData('data', n_test=5, print_options=False)
	# X_train, y_train, X_test, y_test = loadData('pose', n_test=5, print_options=True)
	# X_train, y_train, X_test, y_test = loadData('illum', n_test=5, print_options=True)

	A_mtx = MDA(X_train=X_train, m_dims=10, lam_w_cov = 0.5, print_options = True)
	
	t_X_train = transformData(X_train, A_mtx)
	t_X_test = transformData(X_test, A_mtx)

	if print_options==True:
		print("Transformed data :",t_X_train.shape)
		print("Transformed data :",t_X_test.shape)


if __name__=='__main__':
	main()