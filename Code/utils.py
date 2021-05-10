from scipy.io import loadmat
import numpy as np 
import matplotlib.pyplot as plt


def loadData(dataset_name, n_test, print_options=False):
	if dataset_name == 'pose':
		data_mat = loadmat('../Data/pose.mat')
		data_input = np.moveaxis(data_mat['pose'],-1,0)
		data_input = np.moveaxis(data_input,-1,1)
		data_input = np.reshape(data_input,(68,13,(48*40)))
		data_labels = np.arange(68)

		# # Plot sample data
		# fig,a =  plt.subplots(5,13)
		# for i in range(13):
		# 	for j in range(0,5):
		# 		a[j,i].imshow(np.reshape(data_input[j,i],(48,40)), cmap='gray')
		# 		a[j,i].axis('off')
		# 		a[j,i].set_title(data_labels[j])
		# plt.show()
		
		# Using first 7 rows for training set
		X_train = data_input[:,:7,:]
		y_train = data_labels

		# Random selection from remaining set
		random_idx = np.random.choice(range(X_train.shape[0]), n_test, replace=False)
		random_idy = np.random.randint(7,9)
		X_test = data_input[random_idx,random_idy,:]
		y_test = data_labels[random_idx]

		if print_options==True:
			print("X_train shape :",X_train.shape)
			print("y_train shape :",y_train.shape)
			print("X_test shape :",X_test.shape)
			print("y_test shape :",y_test.shape)

	if dataset_name == 'data':
		data_mat = loadmat('../Data/data.mat')
		data_input = data_mat['face']
		data_input = np.moveaxis(data_input,-1,0)
		data_input = np.reshape(data_input,(600,(24*21)))
		data_input = np.reshape(data_input,(200,3,(24*21)))
		data_labels = np.arange(data_input.shape[0])
		
		# # Plot sample data
		# fig,a =  plt.subplots(5,3)
		# for i in range(3):
		# 	for j in range(0,5):
		# 		a[j,i].imshow(np.reshape(data_input[j,i],(48,40)), cmap='gray')
		# 		a[j,i].axis('off')
		# 		a[j,i].set_title(data_labels[j])
		# plt.show()
		
		# Using first 2 rows for training set
		X_train = data_input[:,:2,:]
		y_train = data_labels

		# Random selection from remaining set
		random_idx = np.random.choice(range(X_train.shape[0]), n_test, replace=False)
		random_idy = 2
		X_test = data_input[random_idx,random_idy,:]
		y_test = data_labels[random_idx]

		if print_options==True:
			print("X_train shape :",X_train.shape)
			print("y_train shape :",y_train.shape)
			print("X_test shape :",X_test.shape)
			print("y_test shape :",y_test.shape)
		
	if dataset_name == 'illum':
		data_mat = loadmat('../Data/illumination.mat')
		data_input = data_mat['illum']
		temp = []
		for i in range(data_input.shape[-1]):
			for j in range(data_input.shape[-2]):
				temp.append(np.reshape(data_input[:,j,i],(40,48)).T)
		temp = np.array(temp)
		temp = np.reshape(temp,(1428,(48*40)))
		temp = np.reshape(temp,(68,21,(48*40)))
		data_input = np.copy(temp)
		data_labels = np.arange(data_input.shape[0])
		
		# # Plot sample dat
		# fig,a =  plt.subplots(5,21)
		# for i in range(5):
		# 	for j in range(21):
		# 		a[i,j].imshow(np.reshape(data_input[i,j],(48,40)), cmap='gray')
		# 		a[i,j].axis('off')
		# 		a[i,j].set_title(data_labels[i])
		# plt.show()
		
		# Using first 16 rows for training set
		X_train = data_input[:,:16,:]
		y_train = data_labels

		# Random selection from remaining set
		random_idx = np.random.choice(range(X_train.shape[0]), n_test, replace=False)
		random_idy = np.random.randint(16,21)
		X_test = data_input[random_idx,random_idy,:]
		y_test = data_labels[random_idx]

		if print_options==True:
			print("X_train shape :",X_train.shape)
			print("y_train shape :",y_train.shape)
			print("X_test shape :",X_test.shape)
			print("y_test shape :",y_test.shape)
		

	return X_train, y_train, X_test, y_test

def transformData(X_train, A_mtx):
	t_X_train = []
	for i in range(X_train.shape[0]):
		t_X_train.append((np.matmul(A_mtx.T,X_train[i].T)).T)
	t_X_train = np.array(t_X_train)
	return t_X_train

def loadExpMDA(n_test = 15,print_options=False):
# X_train, y_train, X_test, y_test = loadData(dataset_name=dataset_name,n_test=n_test, print_options=False)
	data_mat = loadmat('../Data/data.mat')
	data_input = data_mat['face']
	data_input = np.moveaxis(data_input,-1,0)
	data_input = np.reshape(data_input,(600,(24*21)))
	data_input = np.reshape(data_input,(200,3,(24*21)))
	data_input = np.moveaxis(data_input,1,0)

	data_input = data_input[:2,:,:]
	data_label = np.zeros(data_input[:2,:,1].shape, dtype=np.int64)
	data_label[1:]=1

	train_idx = np.arange(0,200-n_test)
	test_idx = np.arange(200-n_test,200)

	random_idx = np.random.choice(np.array([0,1]), n_test)
	X_train = data_input[:,train_idx]
	y_train = data_label[:,train_idx]
	X_test = data_input[random_idx,test_idx]
	y_test = data_label[random_idx,test_idx]

	if print_options==True:
		print("X_train shape :", X_train.shape)
		print("y_train shape :", y_train.shape)
		print("X_test shape :", X_test.shape)
		print("y_test shape :", y_test.shape)
	return X_train, y_train, X_test, y_test


def loadExpData(dataset_name = 'data', cross_val_idx = 0, print_options=False):
	if dataset_name == 'data':
		# X_train, y_train, X_test, y_test = loadData(dataset_name=dataset_name,n_test=n_test, print_options=False)
		data_mat = loadmat('../Data/data.mat')
		data_input = data_mat['face']
		data_input = np.moveaxis(data_input,-1,0)
		data_input = np.reshape(data_input,(600,(24*21)))
		data_label = np.ones(data_input.shape[0], dtype=np.int64)
		data_label[1::3]=-1

		data_input = np.delete(data_input,np.arange(2,data_input.shape[0],3), axis=0)
		data_label = np.delete(data_label,np.arange(2,data_label.shape[0],3))
		
		if cross_val_idx == 0:
			test_idx = np.arange(300,400)		
		if cross_val_idx == 1:
			test_idx = np.arange(200,300)
		if cross_val_idx == 2:
			test_idx = np.arange(100,200)
		if cross_val_idx == 3:
			test_idx = np.arange(0,100)

		tot_idx = np.arange(0,400)
		train_idx = np.setdiff1d(tot_idx,test_idx)

		X_train = data_input[train_idx]
		y_train = data_label[train_idx]
		X_test = data_input[test_idx]
		y_test = data_label[test_idx]
		
		# fig,a =  plt.subplots(6,2)
		# for i in range(6):
		# 	a[i,0].imshow(np.reshape(X_train[2*i+0],(24,21)), cmap='gray')
		# 	a[i,0].axis('off')
		# 	a[i,0].set_title(y_train[2*i+0])
		# 	a[i,1].imshow(np.reshape(X_train[2*i+1],(24,21)), cmap='gray')
		# 	a[i,1].axis('off')
		# 	a[i,1].set_title(y_train[2*i+1])
		# plt.show()

		if print_options==True:
			print("X_train shape :", X_train.shape)
			print("y_train shape :", y_train.shape)
			print("X_test shape :", X_test.shape)
			print("y_test shape :", y_test.shape)
		return X_train, y_train, X_test, y_test

