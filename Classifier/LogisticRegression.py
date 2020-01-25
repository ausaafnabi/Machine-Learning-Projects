import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt

DATASET_PATH = os.path.join("../","datasets")
csv_file_path = os.path.join(DATASET_PATH,"ChurnData.csv")

churn_df = pd.read_csv(csv_file_path)
print(churn_df.head())

churn_df = churn_df[['tenure','age','address','income','ed','employ','equip','callcard','wireless','churn']]
churn_df['churn'].astype('int')
print(churn_df.shape)

X = np.asarray(churn_df[['tenure','age','address','income','ed','employ','equip','callcard','wireless']])
print(X[0:5])

y = np.asarray(churn_df['churn'])
print(y[0:5])

X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 4)
print('Train set : ',X_train.shape,y_train.shape)
print('Test Set : ',X_test.shape,y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01,solver='liblinear').fit(X_train,y_train)
print(LR)

yhat = LR.predict(X_test)
print(yhat)

yhat_proba = LR.predict_proba(X_test)
print(yhat_proba)


#### Evaluation#########

from sklearn.metrics import jaccard_score
jac= jaccard_score(y_test,yhat)
print(jac)

####Confusion Matrics#########

from sklearn.metrics import classification_report,confusion_matrix
import itertools

def plot_confusion_matrix(cm,classes,normalize = False,title='ConfusinMatrix',cmap=plt.cm.Blues):
	"""
	{ function_description }

	:param      cm:         The centimeters
	:type       cm:         { type_description }
	:param      classes:    The classes
	:type       classes:    { type_description }
	:param      normalize:  The normalize
	:type       normalize:  boolean
	:param      title:      The title
	:type       title:      string
	:param      cmap:       The cmap
	:type       cmap:       { type_description }
	"""
	
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
		print('Normalized confusion_matrix')
	else:
		print('Confusion Matrix, without normalization')

	print(cm)

	plt.imshow(cm,interpolation='nearest',cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks,classes,rotation=45)
	plt.yticks(tick_marks,classes)

	fmt = '.2f' if  normalize else 'd'
	thresh = cm.max() / 2.
	for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
		plt.text(j,i,format(cm[i,j],fmt),
			horizontalalignment='center',
			color='white' if cm[i,j] > thresh else 'black')

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('predict label')
print(confusion_matrix(y_test,yhat,labels=[1,0]))

cnf_matrix = confusion_matrix(y_test,yhat,labels=[1,0])
np.set_printoptions(precision = 2)

## plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['churn=1','churn=0'],normalize= False,title="confusion_matrix")
plt.show()
print(classification_report(y_test,yhat))

from sklearn.metrics import log_loss
print('Log Loss:',log_loss(y_test,yhat_proba))