#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.bigdatauniversity.com"><img src="https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width="400" align="center"></a>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>

# In this notebook we try to practice all the classification algorithms that we learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:

# In[130]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |

# Lets download the dataset

# In[131]:

get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File  

# In[132]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[133]:


df.shape


# ### Convert to date time object 

# In[134]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 
# 

# Let’s see how many of each class is in our data set 

# In[135]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection 
# 

# Lets plot some columns to underestand data better:

# In[136]:


# notice: installing seaborn might takes a few minutes
#!conda install -c anaconda seaborn -y


# In[137]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[138]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[139]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[140]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[141]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[142]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding  
# #### How about education?

# In[143]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[144]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

# In[145]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[146]:


X = Feature
X[0:5]


# What are our lables?

# In[147]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[148]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# 
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.

# ### Spliting Dataset into Test and Train

# In[149]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 4)
print('Train set : ',X_train.shape,y_train.shape)
print('Test Set : ',X_test.shape,y_test.shape)


# # Importing Metrics

# In[150]:


from sklearn import metrics
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score


# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[151]:


from sklearn.neighbors import KNeighborsClassifier


# In[152]:


KN = 40
meanAccuracy = []
stdAccuracy = []
for n in range(1,KN):
    
    #Train Model and Predict :: KNN  
    neighbor = KNeighborsClassifier(n_neighbors = n, weights='distance').fit(X_train,y_train)
    yhat=neighbor.predict(X_test)
    meanAccuracy.append(metrics.accuracy_score(y_test, yhat))
    stdAccuracy.append(np.std(yhat==y_test)/np.sqrt(yhat.shape[0]))

mean_accuracy = np.array(meanAccuracy)
std_accuracy = np.array(stdAccuracy)

plt.plot(range(1,KN),mean_accuracy,'g')
plt.fill_between(range(1,KN),mean_accuracy - 1 * std_accuracy,mean_accuracy + 1 * std_accuracy, alpha=0.20)
plt.legend((r'Accuracy ', 'FillRange'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# In[153]:


print( "The best accuracy was with", np.round(mean_accuracy.max(),2), "with k=", mean_accuracy.argmax()+1) 


# # Decision Tree

# In[154]:


from sklearn.tree import DecisionTreeClassifier


# In[155]:


KN = 40
meanAccuracy = []
stdAccuracy = []
for n in range(3,KN):
    
    #Decision Tree 
    dataset_Tree = DecisionTreeClassifier(criterion = "entropy", max_depth=n).fit(X_train,y_train)
    predTree=dataset_Tree.predict(X_test)
    meanAccuracy.append(metrics.accuracy_score(y_test, predTree))
    stdAccuracy.append(np.std(yhat==y_test)/np.sqrt(yhat.shape[0]))

mean_accuracy = np.array(meanAccuracy)
std_accuracy = np.array(stdAccuracy)


plt.figure()
plt.plot(range(3,KN),mean_accuracy,'g')
plt.fill_between(range(3,KN),mean_accuracy - 1 * std_accuracy,mean_accuracy + 1 * std_accuracy, alpha=0.10)
plt.legend((r'Accuracy ', 'DepthRange'))
plt.ylabel('Accuracy ')
plt.xlabel('Max Depth')
plt.tight_layout()
plt.show()


# In[156]:


print( "The best accuracy was with", np.round(mean_accuracy.max(),2), "with depth=", mean_accuracy.argmax()+1) 


# # Support Vector Machine

# In[157]:


from sklearn import svm


# In[158]:


Kernels = ['sigmoid','poly','linear','rbf']
meanAccuracy = []
stdAccuracy = []
for (i,j) in enumerate(Kernels):
    
    #Support Vector Machine 
    SVMClf = svm.SVC(kernel = j, gamma='auto').fit(X_train,y_train)
    yhat = SVMClf.predict(X_test)
    meanAccuracy.append(metrics.accuracy_score(y_test, yhat))
    stdAccuracy.append(np.std(yhat==y_test)/np.sqrt(yhat.shape[0]))

mean_accuracy = np.array(meanAccuracy)
std_accuracy = np.array(stdAccuracy)

plt.plot(np.arange(len(Kernels))+1,mean_accuracy,'g')
plt.fill_between(np.arange(len(Kernels))+1,mean_accuracy - 1 * std_accuracy,mean_accuracy + 1 * std_accuracy, alpha=0.10)
plt.legend((r'Accuracy ', 'ARange'))
plt.ylabel('Accuracy ')
plt.xlabel('Kernels')
plt.tight_layout()
plt.show()


# # Logistic Regression

# In[159]:


from sklearn.linear_model import LogisticRegression


# In[186]:


KN = np.linspace(.01,.1,100)
meanAccuracy = []
stdAccuracy = []
for n in range(1,KN.shape[0]):
    LR = LogisticRegression(C=KN[n-1], solver='liblinear').fit(X_train,y_train)
    #Training Linear Regression Model 
    yLhat = LR.predict(X_test)
    meanAccuracy.append(metrics.accuracy_score(y_test, yLhat))
    stdAccuracy.append(np.std(yLhat==y_test)/np.sqrt(yLhat.shape[0]))
mean_accuracy = np.array(meanAccuracy)
std_accuracy = np.array(stdAccuracy)
    

plt.plot(KN[1:],mean_accuracy,'g')
plt.fill_between(KN[1:],mean_accuracy - 1 * std_accuracy,mean_accuracy + 1 * std_accuracy, alpha=0.10)
plt.legend((r'Accuracy ', 'Regression Range'))
plt.ylabel('Accuracy ')
plt.xlabel('C')
plt.tight_layout()
plt.show()
print( "The best accuracy was with c = .04" ) 


# # Model Evaluation using Test set

# First, download and load the test set:

# In[169]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation 

# In[170]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[177]:


test_df = pd.read_csv('loan_test.csv')
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek

test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_Feature = test_df[['Principal','terms','age','Gender','weekend']]
test_Feature = pd.concat([test_Feature,pd.get_dummies(test_df['education'])], axis=1)
test_Feature.drop(['Master or Above'], axis = 1,inplace=True)
test_Feature.head()
X_train = Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)


# In[179]:


X_train = Feature
y_train = df['loan_status'].values
scaler = preprocessing.StandardScaler().fit(X_train)
scalerX_train = scaler.transform(X_train)

X_test = test_Feature
y_test = test_df['loan_status'].values
scalerX_test= scaler.transform(X_test)


# In[210]:


#using best predictions of the model
models = {
    'KNN': KNeighborsClassifier(n_neighbors = 36, weights='distance'),
    'Decision Tree': DecisionTreeClassifier(criterion="entropy", max_depth = 4),
    'SVM': svm.SVC(kernel='linear'),
    'Logistic Regression': LogisticRegression(C=0.05, solver='liblinear'),
}


# In[211]:


for (kind,model) in models.items():
    model.fit(scalerX_train,y_train)
for (kind,model) in models.items():
    ypred = model.predict(scalerX_test)
    jaccard = round(jaccard_similarity_score(y_test,ypred),2)
    f1 = metrics.f1_score(y_test,ypred,average='weighted')
    if kind == 'Logistic Regression':
        logloss = metrics.log_loss(y_test,model.predict_proba(scalerX_test),normalize=True)
    else:
        logloss = np.nan
    print(f'| {kind:30s} | {jaccard:.3f} | {f1:.3f} | {logloss:.3f} |');


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | 0.778   | 0.755    | NA      |
# | Decision Tree      | 0.759   | 0.696    | NA      |
# | SVM                | 0.741   | 0.630    | NA      |
# | LogisticRegression | 0.759   | 0.672    | .492    |

# 
# 
# <p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>

# In[ ]:




