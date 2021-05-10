# Introduction
This  project  is  a  demonstration  of  the  different  machinelearning  algorithms  for  applications  on  facial  datasets  pro-vided.  There  are  two  main  classification  problems  that  aresolved addressed in this project - face recognition and binaryclassification  of  face  expressions.  An  Kernel  SVM  and  aboosted  SVM  algorithms  were  used  for  the  task  of  binaryclassification  of  facial  expressions.  While  for  the  task  of  thefacial recognition is done using a Maximum Likelihood BayesClassifier  and  a  K-Nearest  neighbor  Classifier.  While  theseconventional  machine  learning  algorithms  perform  well  withsmaller dimension inputs, it is more often than not to get goodoutputs with high dimensional data like in the case of images.Results were compared after performing dimensionality reduc-tion techniques like MDA and PCA.

# Directory Structure
The directory structure for this project is 

```
.
├── Code
│   ├── bayes.py
│   ├── boosted_svm.py
│   ├── knn.py
│   ├── MDA.py
│   ├── PCA.py
│   ├── __pycache__
│   │   ├── MDA.cpython-37.pyc
│   │   ├── PCA.cpython-37.pyc
│   │   ├── svm.cpython-37.pyc
│   │   └── utils.cpython-37.pyc
│   ├── svm.py
│   └── utils.py
├── Data
│   ├── data.mat
│   ├── illumination.mat
│   ├── pose.mat
│   └── README
├── README.md
├── Report.pdf
└── Results
    ├── Boosted_SVM_Training.png
    ├── eigenfaces_data.png
    ├── eigenfaces_illum.png
    ├── eigenfaces_pose.png
    ├── MDA_eigenfaces_data.png
    ├── MDA_eigenfaces_illum.png
    ├── MDA_eigenfaces_pose.png
    ├── MDA_Exp.png
    └── PCA_Exp.png

4 directories, 26 files

```

# Dependencies

- Python==3.7
- Matplotlib 
- cvxopt
- scipy
- numpy

# Dataset
There  are  3  datasets  used  for  analysis  namelyData,  Poseand  Illumination.  The  data  sets  all  contain  the  labelled  im-ages  of  the  people,  with  expressions,  different  expressionsand  in  different  illuminations  respectively.  The  datasets  areread  using  the  functions  mentioned  in  theutils.pyscript  inthe  code.  The  data  format  for  the  face  recognition  task  is [labels,featuredimages,imwidth×imheight],  while  forfacial recognition SVM the data format is[labels,imwidth×imheight].  TheDatadataset  contains  faces  of  200  faces  of3 images features for each person, thePosedataset containsfaces of 68 people with 13 different poses, and the illuminationdata consists faces of 68 people with 21 different illuminatedfaces.

# Installation instructions
All the dependencies can be installed using pip wheel installer. 
```
pip3 install matplotlib
pip3 install cvxopt
pip3 install numpy
pip3 install scikit-learn
```

# Run instructions
Open the `.py` file you want to run. All `.py` files except `utils.py` have a main() method implementation. The different variations of the usage are also mentioned commented in between with instructions. Just uncomment the required implementation and set `print_options` argument to `True` wherever you want to see detailed printout.

# Code Details

## `utils.py`

Contains all utility functions to load the dataset and transform it for dimensionality reduction.

- loadData(dataset_name, n_test, print_options=False)
	- Load the dataset for face recognition task
	- dataset_name - string - name of the dataset ('data', 'pose', 'illum')
	- n_test - int - number of test images

- transformData(X_train, A_mtx)
	- Used to transform the given data using the A_mtx
	- X_train - numpy array -  Data to be transposed
	- A_mtx - numpy array - matrix from PCA/MDA

- loadExpMDA(n_test = 15,print_options=False)
	- Loads the face expression vs neutral face data for Bayes, KNN, PCA, MDA.
	- n_test - int - number of test samples

 - loadExpData(dataset_name = 'data', cross_val_idx = 0, print_options=False)
 	- Load face expression vs neutral face data for SVM and boosted SVM
 	- cross_val_idx - int in range [0,3] - index of the test set for k-fold cross validation

## `PCA.py`

Implementation of the Principal Component Analysis function

- PCA(X_train, alpha, print_options=False)
	- X_train - numpy array - input dataset
	- alpha - float - error coefficient

- Example usage available in main() method. Just load the dataset you want to see results by uncomment the corresponding loadData() Method and set all `print_options` arguments to true.

 ![alt text](./Results/eigenfaces_data.png?raw=true "PCA Example on Data Dataset")


## `MDA.py`

Implementation of the Multiple Discriminant Analysis function

- MDA(X_train, m_dims, lam_w_cov=0.2 , print_options = False)
	- X_train - numpy array - input dataset
	- lam_w_cov - float - regularization coefficient for sigma_w

- Example usage available in main() method. Just load the dataset you want to see results by uncomment the corresponding loadData() Method and set all `print_options` arguments to true.

 ![alt text](./Results/MDA_eigenfaces_data.png?raw=true "MDA Example on Data Dataset")


## `bayes.py`

Implementation of the maximum likelihood bayes classifier on all the 3 datasets. 

- bayesClassifier(X_train, lam, print_options=False)
	- A method to find the maximum likelihood estimates of the classes
	- X_train - numpy array - input dataset
	- lam - float - regularization coefficient for covariance matrix

- evalClassifier(X_train,mean_ML,sigma_ML,X_test,y_test,print_options=False)
	- Evaluates the accuracy of the bayes classifier on the set of testing samples.
	- X_train - numpy array - input dataset
	- mean_ML, sigma_ML - numpy array - return vales of ML estimator 
	- X_test, y_test - test dataset

- Example usage available in main() method. Just load the dataset you want to see results by uncomment the corresponding loadData() Method and set all `print_options` arguments to true. Code can run woth or without dimensionality reduction for both the tasks. Just comment the lines corresponding to dimensionality reduction.

## `knn.py`
Implementation of the K-nearest neighbors classifier

- KNN(X_train,k, x, print_options=False)
	- Method to find the label based on k- nearest neighbor labels
	- X_train - input dataset
	- k - int - K values
	- x - np array - test image

- evalClassifier(X_train,k ,X_test,y_test,print_options=False)
	- Method to find the label based on k- nearest neighbor labels
	- X_train - input dataset
	- k - int - K values
	- X_test, y_test - test dataset

- Example usage available in main() method. Just load the dataset you want to see results by uncomment the corresponding loadData() Method and set all `print_options` arguments to true. Code can run woth or without dimensionality reduction for both the tasks. Just comment the lines corresponding to dimensionality reduction.


## `svm.py`

Implementation for the SVM classifier (Expression vs Neutral Classification)
- SVM Class Object
	- class with some attributes necessary for computing support vectors
	
	- fit(X_train,y_train, print_options=False)
		- Method to compute the support vectors by solving dual problem.

	- eval(X_test, y_test)
		- Method to evaluate the SVM classifier on the test data

	- predict(X_test)
		- Method to predict the values of any new image on the trained classifier.

- Example usage available in main() method. Just load the dataset you want to see results by uncomment the corresponding loadData() Method and set all `print_options` arguments to true. Code can run woth or without dimensionality reduction for both the tasks. Just comment the lines corresponding to dimensionality reduction.


## `boosted_svm.py`

Implementation of the Adaboost SVM classifier. Uses the SVM class to find the weak classifiers. 

- Example usage available in main() method. Just load the dataset you want to see results by uncomment the corresponding loadData() Method and set all `print_options` arguments to true. Code can run woth or without dimensionality reduction for both the tasks. Just comment the lines corresponding to dimensionality reduction.

