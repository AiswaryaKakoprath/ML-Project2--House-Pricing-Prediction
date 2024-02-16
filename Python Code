#!/usr/bin/env python
# coding: utf-8

# # Project: Investement Prediction
# * Objective: study the investment pattern of bank customers to predict whether a new customer will invest or not

# In[1]:


#for data preparation
import pandas as pd

#for plotting for eda
import matplotlib.pyplot as plt
import seaborn as sns

#for data sampling
from sklearn.model_selection import train_test_split

#for model building
from sklearn.linear_model import LogisticRegression

#for confusion matrix, accuracy, precision, and recall
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[2]:


#import data
df=pd.read_csv(r"C:\Users\aksha\OneDrive\Desktop\Introtallent\python\Data Files used in Projects\Investment.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


#Replace Yes wiht 1 and No with 0 in Invested column
df['Invested']=df['Invested'].replace(['Yes','No'],[1,0])


# In[6]:


df.head()


# In[7]:


#dtypes
df.dtypes


# In[8]:


#check missing values
df.isnull().sum() #No missing values


# In[9]:


#Outliers
plt.boxplot(df['age']) # has outliers
plt.show()


# In[10]:


#Outliers
plt.boxplot(df['cons_conf_idx']) # has outliers
plt.show()


# In[11]:


#Outliers
plt.boxplot(df['emp_var_rate']) # no outliers
plt.show()


# In[12]:


#Outliers
plt.boxplot(df['euribor3m']) # no outliers
plt.show()


# In[13]:


#user defined function for outlier treatment
def remove_outlier(d,c):
    q1=d[c].quantile(0.25)
    q3=d[c].quantile(0.75)
    
    iqr=q3-q1
    
    ub=q3+1.5*iqr
    lb=q1-1.5*iqr
    
    output=d[(d[c]>lb) & (d[c]<ub)]
    return output


# In[14]:


#remove outliers from Age and cons_conf_idx
df=remove_outlier(df,'age')
plt.boxplot(df['age'])
plt.show()


# In[15]:


df=remove_outlier(df,'cons_conf_idx')
plt.boxplot(df['cons_conf_idx'])
plt.show()


# # Check values in categorical variable to ensure there is no issue

# In[17]:


df['job'].unique()


# In[18]:


df['marital'].unique()


# In[19]:


df['education'].unique()


# In[20]:


#replace basic.4y, basic.6y, and basic.9y with 'basic'
df['education']=df['education'].replace(['basic.4y','basic.6y','basic.9y'],['basic','basic','basic'])


# In[21]:


df['education'].unique()


# In[22]:


df['default'].unique()


# In[23]:


df['housing'].unique()


# In[24]:


df['loan'].unique()


# In[25]:


df['loan'].unique()


# In[26]:


df['contact'].unique()


# In[27]:


df['month'].unique()


# In[28]:


df['day_of_week'].unique()


# In[29]:


df['poutcome'].unique()


# # -----------EDA Starts----------

# In[33]:


#all plots
sns.pairplot(df)


# # Distribution

# In[32]:


sns.distplot(df['age'])
plt.show()


# In[35]:


sns.distplot(df['duration'])
plt.show()


# In[16]:


sns.distplot(df['emp_var_rate'])
plt.show()


# # Data mix

# In[17]:


df.groupby('Invested')['Invested'].count().plot(kind='bar')
#imbalaced class


# In[18]:


#correlation plot
df_numeric=df.select_dtypes(include=['int64','float64'])
df_numeric.head()


# In[19]:


#remove categorical variable
df_numeric=df_numeric.drop(['campaign','pdays','previous'],axis=1)


# In[20]:


df_numeric.head()


# In[21]:


#heatmap
sns.heatmap(df_numeric.corr(),cmap='YlGnBu',annot=True)


# In[ ]:


#drop unwanted columns
#col_to_drop=['contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays','previous', 'poutcome']
#df=df.drop(col_to_drop, axis=1)


# In[22]:


#one - hot encoding
df_categorical=df.select_dtypes(include='object')
df_categorical


# In[ ]:


get_ipython().run_line_magic('pinfo', 'dummies')
the first column represent the absence of any category and is redundant because the absence


# In[23]:


#create dummies
df_dummies= pd.get_dummies(df_categorical,drop_first=True)
df_dummies.head()


# In[24]:


#create final data
df_final=pd.concat([df_numeric,df_dummies],axis=1)
df_final.head()


# In[25]:


#create x and y
x=df_final.drop('Invested',axis=1)
y=df_final['Invested']


# In[26]:


#check x and y
print(x.shape,y.shape)


# # Feature Selection

# In[27]:


#create traininga nd test samples
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=999)


# In[28]:


#check sample size
print(xtrain.shape,xtest.shape,ytrain.shape,ytest.shape)


# # Feature selection using chi-square test

# In[29]:


#for check the categorical variable/Feature selection using chi-square test
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

key_features = SelectKBest(score_func=f_classif, k=4) 
#to select 5 significant features

# Fit the key_features to the training data and transform it
xtrain_selected = key_features.fit_transform(xtrain, ytrain)

# Get the indices of the selected features
selected_indices = key_features.get_support(indices=True)

# Get the names of the selected features
selected_features = xtrain.columns[selected_indices]


# In[ ]:


#deleting or eliminating the non significant variable-diamensionality reduction
#when we ve too features in this data the curse of diamensionality then do diamensionality reduction


# In[30]:


selected_indices


# In[31]:


#create x train based on selected features
x_train=xtrain[selected_features]


# In[32]:


x_train.columns


# In[33]:


#store KBest columns from xtest to x_test
x_test=xtest[selected_features]


# In[34]:


#Create x_train based on selected features
x_train=xtrain[selected_features]
x_train.columns


# In[ ]:





# # Logistic regression algorithm

# In[35]:


#instantiate Logistic regression
logreg=LogisticRegression()


# # Model 1: Build a model using all features

# In[37]:


#train the model
logreg.fit(xtrain,ytrain)


# In[38]:


#check training accuracy
logreg.score(xtrain,ytrain)


# In[39]:


y_pred=logreg.predict(xtest)
#check prediction accuracy
logreg.score(xtest,ytest)
#accuracy= 0.9113149847094801


# # Model 2: using selected K(4) Best Variables
# * Build a model with 5 variables,accuracy was 0.91
# * Build another model with 4 variables,accuracy was 0.90 hence continued with k=4

# In[41]:


#train the model using xtrain and ytrain (fit the model)
logreg.fit(x_train,ytrain)


# In[42]:


logreg.score(x_train,ytrain)


# In[48]:


#predict investment using test data
lr_predicted=logreg.predict(x_test)


# In[45]:


logreg.score(x_test,ytest)


# In[49]:


#print confusino matrix
confusion_matrix(ytest, lr_predicted)


# In[ ]:





# # Score to Model performance 
# * Accuracy=(TN+TP)/(TN+TP+FN+FP)
# 
# * Precision=TP/(TP+FP)
# 
# * Recall=TP/(TP+FN)
# 
# * F-score=2xPrecisionxRecall/(Precision+Recall)

# In[50]:


#store logistic regression scores in seperate variables
lr_accuracy=accuracy_score(ytest, lr_predicted)
lr_precision=precision_score(ytest, lr_predicted)
lr_recall=recall_score(ytest, lr_predicted)
lr_fscore=f1_score(ytest, lr_predicted)


# In[51]:


col=['Model','Accuracy',  'Precision','Recall', 'F1-Score']
data=[["Log Reg", lr_accuracy,lr_precision, lr_recall,lr_fscore]]
ml_summary=pd.DataFrame(data, columns=col)
ml_summary


# # KNN(K- nearest neighbours) Algorithm
# * k is number of neighbours

# In[53]:


#import knn library from sklearn
from sklearn.neighbors import KNeighborsClassifier


# In[54]:


#create model object
knn=KNeighborsClassifier(n_neighbors=5,metric='euclidean')


# In[55]:


#fit knn model
knn.fit(x_train,ytrain)


# In[56]:


#predict y using knn
knn_predicted=knn.predict(x_test)


# In[57]:


#check scores
knn_accuracy=accuracy_score(ytest,knn_predicted)
knn_precision=precision_score(ytest,knn_predicted)
knn_recall=recall_score(ytest,knn_predicted)
knn_fscore=f1_score(ytest,knn_predicted)


# In[58]:


#update accuracy tables
col=['Model','Accuracy',  'Precision','Recall', 'F1-Score']
data=[["Log Reg", lr_accuracy,lr_precision, lr_recall,lr_fscore],
     ["KNN", knn_accuracy,knn_precision, knn_recall,knn_fscore]]
ml_summary=pd.DataFrame(data, columns=col)
ml_summary


# # Naive Bayes Algorithm

# In[59]:


#import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB


# In[60]:


#create a Gaussian classifier
gnb=GaussianNB()


# In[61]:


#train the model using the training sets
gnb.fit(x_train,ytrain)


# In[62]:


#predict output
gnb_predicted=gnb.predict(x_test)


# In[63]:


#check scores
gnb_accuracy=accuracy_score(ytest, gnb_predicted)
gnb_precision=precision_score(ytest, gnb_predicted)
gnb_recall=recall_score(ytest, gnb_predicted)
gnb_fscore=f1_score(ytest, gnb_predicted)


# In[64]:


# update accuracy tables
col=['Model','Accuracy',  'Precision','Recall', 'F1-Score']
data=[["Log Reg", lr_accuracy,lr_precision, lr_recall,lr_fscore],
     ["KNN", knn_accuracy,knn_precision, knn_recall,knn_fscore],
     ["Naive Bayes", gnb_accuracy,gnb_precision, gnb_recall,gnb_fscore]]
ml_summary=pd.DataFrame(data, columns=col)
ml_summary


# # Decision Tree Classifier

# In[65]:


#import Decision Tree classifier library from sklearn
from sklearn.tree import DecisionTreeClassifier


# In[66]:


#Create model object
dtree=DecisionTreeClassifier(max_depth=5)


# In[67]:


#fit the training model
dtree.fit(xtrain,ytrain)


# In[68]:


#test the model using xtest
dtree_predicted=dtree.predict(xtest)


# In[70]:


#check scores
dtree_accuracy=accuracy_score(ytest, dtree_predicted)
dtree_precision=precision_score(ytest, dtree_predicted)
dtree_recall=recall_score(ytest, dtree_predicted)
dtree_fscore=f1_score(ytest, dtree_predicted)


# In[72]:


#update accuracy tables
col=['Model','Accuracy',  'Precision','Recall', 'F1-Score']
data=[["Log Reg", lr_accuracy,lr_precision, lr_recall,lr_fscore],
     ["KNN", knn_accuracy,knn_precision, knn_recall,knn_fscore],
     ["Naive Bayes", gnb_accuracy,gnb_precision, gnb_recall,gnb_fscore],
     ["Decision Tree", dtree_accuracy,dtree_precision, dtree_recall,dtree_fscore]]
ml_summary=pd.DataFrame(data, columns=col)
ml_summary


# # Random Forest Classifier

# In[73]:


#import Random Forest classifier library from sklearn
from sklearn.ensemble import RandomForestClassifier


# In[74]:


# creating a RF classifier
rfc = RandomForestClassifier(n_estimators = 100) 


# In[75]:


rfc.fit(xtrain,ytrain)


# In[76]:


rfc_predicted=rfc.predict(xtest)


# In[77]:


#check scores
rfc_accuracy=accuracy_score(ytest, rfc_predicted)
rfc_precision=precision_score(ytest, rfc_predicted)
rfc_recall=recall_score(ytest, rfc_predicted)
rfc_fscore=f1_score(ytest, rfc_predicted)


# In[78]:


# update accuracy tables
col=['Model','Accuracy',  'Precision','Recall', 'F1-Score']
data=[["Log Reg", lr_accuracy,lr_precision, lr_recall,lr_fscore],
     ["KNN", knn_accuracy,knn_precision, knn_recall,knn_fscore],
     ["Naive Bayes", gnb_accuracy,gnb_precision, gnb_recall,gnb_fscore],
     ["Decision Tree", dtree_accuracy,dtree_precision, dtree_recall,dtree_fscore],
     ["Random Forest", rfc_accuracy,rfc_precision, rfc_recall,rfc_fscore]]
ml_summary=pd.DataFrame(data, columns=col)
ml_summary


# # SVM (Support Vector Machine) Algorithm

# In[79]:


#Support Vector Machine (SVC: Support Vector Classifier)
from sklearn.svm import SVC  


# In[80]:


#create an instance of SVM
svm = SVC(kernel='linear') 


# In[81]:


#Fit the model
svm.fit(x_train,ytrain)


# In[82]:


#Predict the response from xtest
svm_predicted=svm.predict(x_test)


# In[83]:


#check scores
svm_accuracy=accuracy_score(ytest, svm_predicted)
svm_precision=precision_score(ytest, svm_predicted)
svm_recall=recall_score(ytest, svm_predicted)
svm_fscore=f1_score(ytest, svm_predicted)


# In[85]:


# update accuracy tables
col=['Model','Accuracy',  'Precision','Recall', 'F1-Score']
data=[["Log Reg", lr_accuracy,lr_precision, lr_recall,lr_fscore],
     ["KNN", knn_accuracy,knn_precision, knn_recall,knn_fscore],
     ["Naive Bayes", gnb_accuracy,gnb_precision, gnb_recall,gnb_fscore],
     ["Decision Tree", dtree_accuracy,dtree_precision, dtree_recall,dtree_fscore],
     ["Random Forest", rfc_accuracy,rfc_precision, rfc_recall,rfc_fscore],
     ["SVM", svm_accuracy,svm_precision, svm_recall,svm_fscore]]
ml_summary=pd.DataFrame(data, columns=col)
ml_summary


# In[ ]:


#implementtion


# In[ ]:


#import new data
new_df=pd.read_excel()


# In[ ]:


#check missing values and impute


# In[ ]:


#dummy conversion


# In[ ]:


#createa new_final by combining dummy and numeric


# In[ ]:


#predict y by using the best model
predict_investment=dtree.predict(new_final)


# In[ ]:




