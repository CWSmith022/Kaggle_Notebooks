#!/usr/bin/env python
# coding: utf-8

# # <u> Death Prediction by Heart Failure</u>
# <u> By: Christopher Smith https://github.com/CWSmith022/Learning.git</u>
# 
# The published data is from: <u> Davide Chicco, Giuseppe Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020). </u>
# 
# The .csv file was obtained from: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data
# 
# 
# Heart disease is the leading cause of death for people in the United States. The development of models for prediction for potential of heart disease related death could be important for saving lives. Here, an approach using tools in the sci-kit learn library will be used for prediction of deaths by heart attacks. The process starts by feature selection with the KBestFunction by $Chi^{2}$ score, then the data is preprocessed to be used for several supervised Machine Learning Algorithms.
# 
# The Algorithms used are:
# 
# - Logistic Regression
# - Support Vector Machines
# - K-Nearest Neighbors
# - Random Forest
# - Gradient Boosting
# - Ridge Classifier
# 
# ## <u> Logistic Regression </u> 
# A model that is used statistically for binary dependent variables based on the probability of an event occuring. This can be further extended for several variables in a classification setting for multi-class prediction.
#     
# ## <u> Support Vector Machines (SVM) </u>
# Commonly used for classification tasks, SVM's function by a Kernel which draws points on a hyperplane and uses a set of vectors to separate data points. This separation of data points creates a decision boundary for where a new data point can be predicted for a specific class label. 
# 
# ## <u> K-Nearest Neighbors </u>
# Simply, an algorithm that clusters the data and by a measure of distance to the 'k' nearest points votes for a specific class prediction.
# 
# ## <u> Random Forest </u> 
# An ensemble method that estimates several weak decision trees and combines the mean to create an uncorrelated forest at the end. The uncorrelated forest should be able to predict more accurately than an individual tree.
# 
# ## <u> Gradient Boosting </u>
# Similar to Random Forest, Gradient Boosting builds trees one at a time then ensembles them as each one is built.
# 
# ## <u> Ridge Classifier </u>
# Normalizes data then treats problem as a multi-output regression task.
# 

# ## Table of Contents
# 
# [1.Importing Libraries](#1) <br/>
# [2.Importing Data](#2) <br/>
# [3.Exploring Data](#3) <br/>
# [4.Feature Selection](#3) <br/>
# [5.Splitting the Data](#4) <br/>
# [6.Feature Scaling (Normalization)](#5) <br/>
# [7.Machine Learning](#6) <br/>
#     [7.1.Logistic Regression](#7.1) <br/>
#     [7.2.Support Vector Machine](#7.2) <br/>
#     [7.3.K-Nearest Neighbor](#7.3) <br/>
#     [7.4.Random Forest](#7.4) <br/>
#     [7.5.Gradient Boosting](#7.5) <br/>
#     [7.6.Ridge Classifier](#7.6) <br/>
# [8.Evaluation of Acuracy](#9) <br/>
# [9.Discussion](#10) <br/>

# <a id="1"></a>
# ## Importing Libraries

# In[21]:


#Simple Data processing
import numpy as np #linear algebra
import pandas as pd # data processing, .csv load

#Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Data Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
import itertools #For Confusion Matrix
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Scaling
from sklearn import preprocessing #For data normalization

# Model Selection
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV # For parameterization and splitting data
from sklearn.metrics import confusion_matrix
from sklearn import metrics # For Accuracy

#Classification Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier


# <a id="2"></a>
# # Importing Data

# In[2]:


heart=pd.read_csv('heart_failure_clinical_records_dataset.csv')
heart


# <a id="3"></a>
# # Exploring Data

# In[3]:


heart.describe()


# In[4]:


heart.info()


# In[5]:


print(heart.columns.unique)


# <a id="4"></a>
# # Feature Selection

# In[6]:


#Separating the data to asses with feature selection 
X_feat=heart[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]
y_feat=heart['DEATH_EVENT']


# In[7]:


#Feature Selection
bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X_feat,y_feat)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_feat.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Factors','Score']  #naming the dataframe columns
print(featureScores.nlargest(5,'Score'))  #print 5 best features


# By using KBest selection with the $Chi^{2}$ scorer that the top 5 Factors that could be related to 'DEATH_EVENT' are shown above and these will be used here on out for prediction of 'DEATH_EVENT'.

# <a id="5"></a>
# # Splitting The Data

# In[8]:


train_accuracy= []
accuracy_list = []
algorithm = []

X_train,X_test,y_train,y_test = train_test_split(heart[['platelets','time','creatinine_phosphokinase','ejection_fraction','age']]
                                                 ,heart['DEATH_EVENT'],test_size=0.2, random_state=0)
print("X_train shape :",X_train.shape)
print("Y_train shape :",y_train.shape)
print("X_test shape :",X_test.shape)
print("Y_test shape :",y_test.shape)


# <a id="6"></a>
# # Feature Scaling (Normalization)

# To remove outlier bias the formula $z=(x-u)/s$ is used first on the training set then applied to the testing set

# In[9]:


scaler_ss=preprocessing.StandardScaler()


# In[10]:


X_train_scaled=scaler_ss.fit_transform(X_train)
X_test_scaled=scaler_ss.transform(X_test)


# # Confusion Matrix

# In[11]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.BuGn):

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# <a id="7"></a>
# # Machine Learning

# <b>Alive is representative of (0) while Death is (1)  </b>

# <a id="7.1"></a>
# ## Logistic Regression

# In[12]:


Log_Reg=LogisticRegression(C=1, class_weight='balanced', dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=1000, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
Log_Reg.fit(X_train_scaled, y_train)
y_reg=Log_Reg.predict(X_test_scaled)
print("Train Accuracy {0:.3f}".format(Log_Reg.score(X_train_scaled, y_train)))
print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_reg)))
cm = metrics.confusion_matrix(y_test, y_reg)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm, classes=['Alive', 'Death'],
                          title='Logistic Regression')
accuracy_list.append(metrics.accuracy_score(y_test, y_reg)*100)
train_accuracy.append(Log_Reg.score(X_train_scaled, y_train))
algorithm.append('Logistic Regression')


# <a id="7.2"></a>
# ## Support Vector Machine

# By using GRIDSearchCV the best kernel will be decided for the model.

# In[13]:


SVC_param={'kernel':['sigmoid','rbf','poly'],'C':[1],'decision_function_shape':['ovr'],'random_state':[0]}
SVC_pol=SVC()
SVC_parm=GridSearchCV(SVC_pol, SVC_param, cv=5)
SVC_parm.fit(X_train_scaled, y_train)
y_pol=SVC_parm.predict(X_test_scaled)
print("The best parameters are ",SVC_parm.best_params_)
print("Train Accuracy {0:.3f}".format(SVC_parm.score(X_train_scaled, y_train)))
print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_pol)))
cm = metrics.confusion_matrix(y_test, y_pol)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm, classes=['Alive', 'Death'],
                          title='SVM')
train_accuracy.append(SVC_parm.score(X_train_scaled, y_train))
accuracy_list.append(metrics.accuracy_score(y_test, y_pol)*100)
algorithm.append('SVM')


# <a id="7.3"></a>
# ## K-Nearest Neighbor

# First we need to select the best value of K for the highest accuracy in the model.

# In[14]:


error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):
    K_NN =KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=i, p=2,
                     weights='distance')
    K_NN.fit(X_train_scaled, y_train)
    pred_i = K_NN.predict(X_test_scaled)
    error.append(np.mean(pred_i != y_test))


# In[15]:


plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# Looking at the error helps decide the best K-Value given the parameters. The lower the error at K the better accuracy there will be.

# In[16]:


K_NN =KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=2, p=2,
                     weights='distance')
K_NN.fit(X_train_scaled, y_train)
y_KNN=K_NN.predict(X_test_scaled)
print("Train Accuracy {0:.3f}".format(K_NN.score(X_train_scaled, y_train)))
print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_KNN)))
cm = metrics.confusion_matrix(y_test, y_KNN)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm, classes=['Alive', 'Death'],
                          title='KNN')
train_accuracy.append(K_NN.score(X_train_scaled, y_train))
accuracy_list.append(metrics.accuracy_score(y_test, y_KNN)*100)
algorithm.append('KNN')


# <a id="7.4"></a>
# ## Random Forest

# By using GRIDSearchCV the best parameters will be decided for the model.

# In[17]:


RFC_param={'max_depth':[1,2,3,4,5],'n_estimators':[10,25,50,100,150],'random_state':[None],'criterion':['entropy','gini'],'max_features':[0.5]}
RFC=RandomForestClassifier()
RFC_parm=GridSearchCV(RFC, RFC_param, cv=5)
RFC_parm.fit(X_train_scaled, y_train)
y_RFC=RFC_parm.predict(X_test_scaled)
print("The best parameters are ",RFC_parm.best_params_)
print("Train Accuracy {0:.3f}".format(RFC_parm.score(X_train_scaled, y_train)))
print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_RFC)))
cm = metrics.confusion_matrix(y_test, y_RFC)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm, classes=['Alive', 'Death'],
                          title='RFC')
train_accuracy.append(RFC_parm.score(X_train_scaled, y_train))
accuracy_list.append(metrics.accuracy_score(y_test, y_RFC)*100)
algorithm.append('Random Forest')


# <a id="7.5"></a>
# ## Gradient Boosting Classifier

# By using GRIDSearchCV the best parameters will be decided for the model.

# In[18]:


GBC_parma={'loss':['deviance','exponential'],'n_estimators':[10,25,50,100,150],'learning_rate':[0.1,0.25, 0.5, 0.75],
          'criterion':['friedman_mse'], 'max_features':[None],'max_depth':[1,2,3,4,5,10],'random_state':[None]}
GBC = GradientBoostingClassifier()
GBC_parm=GridSearchCV(GBC, GBC_parma, cv=5)
GBC_parm.fit(X_train_scaled, y_train)
y_GBC=GBC_parm.predict(X_test_scaled)
print("The best parameters are ",GBC_parm.best_params_)
print("Train Accuracy {0:.3f}".format(GBC_parm.score(X_train_scaled, y_train)))
print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_GBC)))
cm = metrics.confusion_matrix(y_test, y_GBC)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm, classes=['Alive', 'Death'],
                          title='GBC')
train_accuracy.append(GBC_parm.score(X_train_scaled, y_train))
accuracy_list.append(metrics.accuracy_score(y_test, y_GBC)*100)
algorithm.append('GBC')


# <a id="7.6"></a>
# ## Ridge Classifier

# By using GRIDSearchCV the best parameters will be decided for the model.

# In[32]:


RC_parma={'solver':['svd','lsqr','cholesky'],'alpha':[0,0.5,0.75,1,1.5,2],'normalize':[True,False]}
RC=RidgeClassifier()
RC_parm=GridSearchCV(RC, RC_parma, cv=5)
RC_parm.fit(X_train_scaled, y_train)
y_RC=RC_parm.predict(X_test_scaled)
print("The best parameters are ",RC_parm.best_params_)
print("Train Accuracy {0:.3f}".format(RC_parm.score(X_train_scaled, y_train)))
print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_RC)))
cm = metrics.confusion_matrix(y_test, y_RC)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm, classes=['Alive', 'Death'],
                          title='Ridge Classifier')
train_accuracy.append(RC_parm.score(X_train_scaled, y_train))
accuracy_list.append(metrics.accuracy_score(y_test, y_RC)*100)
algorithm.append('Ridge Classifier')


# <a id="9"></a>
# # Evaluation of Accuracy

# In[39]:


#Train Accuracy
f,ax = plt.subplots(figsize = (10,5))
sns.barplot(x=train_accuracy,y=algorithm,palette = sns.dark_palette("blue",len(accuracy_list)))
plt.xlabel("Accuracy")
plt.ylabel("Algorithm")
plt.title('Algorithm Train Accuracy')
plt.show()


# In[43]:


#Testing Accuracy
f,ax = plt.subplots(figsize = (10,5))
sns.barplot(x=accuracy_list,y=algorithm,palette = sns.dark_palette("blue",len(accuracy_list)))
plt.xlabel("Accuracy")
plt.ylabel("Algorithm")
plt.title('Algorithm Test Accuracy')
plt.show()


# <a id="10"></a>
# # Discussion

# - Using the KBest approach with $Chi^{2}$ score can be an effective approach for feature selection.
# - However, other methods for this data set in feature selection should be suggested such as a correlation matrix or tree importance based selection method
# - Training accuracy does not mean the model will predict as well and models with lower training accuracy can predict better
# - Lastly, tree ensembles may be a better selection for this type of data set with the given features used.
# - If this notebook is helpful please provide an upvote!
# 
