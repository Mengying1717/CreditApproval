#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

#load data
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
df = pd.read_table('crx.data', sep=',', header=None, na_values=["?"])
df.head()


# In[381]:


#checking missing data
df.isnull().sum()


# In[ ]:


df = df.dropna(axis = 0)


# In[346]:


df.isnull().sum()


# In[347]:


df.head()


# In[348]:


df_x = df.loc[:, 0: 14]


# In[349]:


df_x.head()


# In[350]:


df_x.shape


# In[351]:


feature_cat = [feature for feature in df_x.columns if df_x[feature].dtypes == 'O']


# In[352]:


feature_num = [feature for feature in df_x.columns if df_x[feature].dtypes != 'O']


# In[353]:


df_dummies = pd.get_dummies(df_x[feature_cat])


# In[354]:


df_dummies.shape


# In[355]:


clean_df = pd.concat([df_x[feature_num],df_dummies],axis=1)


# In[356]:


clean_df.shape


# In[357]:


df_y = df[15]


# In[358]:


df_y.head()


# In[359]:


df_y = df_y.replace('+', 1)
dy_y = df_y.replace('-', 0)


# In[360]:


df_y.unique()[1]


# In[361]:


df_y.replace('-', 0,inplace=True)


# In[362]:


df_y.unique()


# In[363]:


df_y.head()


# In[364]:


clean_df.shape


# In[365]:


df_y.shape


# In[366]:


# split data set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(clean_df, df_y, test_size=0.33, random_state = 42, stratify = df_y)


# In[367]:


x_train


# In[368]:


from sklearn.preprocessing import StandardScaler

scaled_xtrain = x_train.copy()
scaled_xtest = x_test.copy()

scaler = StandardScaler()

scaled_xtrain[feature_num] = scaler.fit_transform(scaled_xtrain[feature_num])
scaled_xtest[feature_num] = scaler.transform(scaled_xtest[feature_num])


# In[369]:


scaled_xtrain


# In[370]:


scaled_xtest


# In[371]:


scaled_xtrain.shape


# In[372]:


y_train.shape


# In[373]:


y_train


# In[374]:


#Train a logistic regression classifier with the class_weight = None
from sklearn.linear_model import LogisticRegression
classificationA = LogisticRegression(penalty = 'none', random_state=42, class_weight = None, solver = 'sag')

classificationA.fit(scaled_xtrain, y_train)


# In[375]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, balanced_accuracy_score
# Use this classifier to predict the train set
y_pred_train = classificationA.predict(scaled_xtrain)
print('The accuracy for model in train set is:', accuracy_score(y_train, y_pred_train))


# In[376]:


# Use this classifier to predict the test set
y_pred_test = classificationA.predict(scaled_xtest)
print('The accuracy for model in test set is:', accuracy_score(y_test, y_pred_test))
print('The confusion_matrix for model in test set is:', confusion_matrix(y_test, y_pred_test))
print('The f1_score for model in test set is:', f1_score(y_test, y_pred_test))


# In[377]:


#Train a logistic regression classifier with the class_weight = balanced
classificationB = LogisticRegression(penalty = 'none', random_state=42, class_weight = 'balanced', solver = 'sag')
classificationB.fit(scaled_xtrain, y_train)


# In[378]:


# Use this classifier to predict the train set
y_pred_balanced_train = classificationB.predict(scaled_xtrain)
print('The accuracy for model in train set is:', accuracy_score(y_train, y_pred_balanced_train))


# In[379]:


# Use this classifier to predict the test set
y_pred_balanced_test = classificationB.predict(scaled_xtest)
print('The accuracy for model in test set is:', accuracy_score(y_test, y_pred_balanced_test))
print('The balanced accuracy for model in test set is:', balanced_accuracy_score(y_test, y_pred_balanced_test))
print('The confusion_matrix for model in test set is:', confusion_matrix(y_test, y_pred_balanced_test))
print('The f1_score for model in test set is:', f1_score(y_test, y_pred_balanced_test))


# In[380]:


# Plot the precision-recall curve and report the Average Precision
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

y_score = classificationB.decision_function(scaled_xtest)
average_precision = average_precision_score(y_test, y_score)
print('Average precision-recall score: {0:0.3f}'.format(
      average_precision))
y_precision, y_recall, _ = precision_recall_curve(y_test, y_score, )
pr_display = PrecisionRecallDisplay(precision = y_precision, recall = y_recall).plot()

