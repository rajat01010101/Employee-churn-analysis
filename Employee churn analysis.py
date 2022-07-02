#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[328]:


#load data 
data=pd.read_csv('HR_com.csv')
data.head()


# In[329]:


#finding the count of missing values from different columns
data.isna().sum()


# In[330]:


data.info() # printing the summary of the dataframe


# In[331]:


data.shape


# In[332]:


# describe data
data.describe()


# In[333]:


data.groupby('salary').mean()  #Now lets have a quick analysis or employees according to their salaries


# In[334]:


data.groupby('Department').mean() #Now lets have a quick analysis or employees according to their Department.


# # Cleaning of data
# 

# In[335]:


#finding the count of missing values from different columns
data.isna().sum()


# In[336]:


# find duplicates values in data
data[data.duplicated()].sum()


# In[337]:


# Drop duplicates 
data=data.drop_duplicates()
data


# In[338]:


data[data.duplicated()].sum()


# # Data visualization

# In[339]:


sns.countplot(data.left,palette='Set1')


# In[340]:


sns.countplot(x='salary',hue='left',palette='Set2',data=data)


# In[341]:


plt.figure(figsize=(15,8))
sns.countplot(x='Department',hue='left',palette='Set2',data=data)


# In[342]:


plt.figure(figsize=(15,8))
sns.countplot(x='Department',hue='number_project',palette='Set3',data=data)


# In[343]:


data.plot.hexbin(x='satisfaction_level',y='last_evaluation',gridsize=4,C='number_project',figsize=(10,5))


# In[344]:


sns.pairplot(data)


# In[345]:


sns.pairplot(data,hue='left')


# In[346]:


plt.figure(figsize = (13, 7))
sns.heatmap(data.corr(), annot = True, cmap = 'Greens')


# #  model  building

# In[347]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:





# In[348]:


data = pd.get_dummies(data, columns=['salary'])
data


# In[349]:


x = data.drop(columns = ['left', 'Department', 'Work_accident'])
y = data['left']


# In[350]:


x


# In[351]:


y


# In[352]:


sc = StandardScaler()
x = sc.fit_transform(x)
x


# # LogisticRegression

# In[353]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)


# In[354]:


logi_model=LogisticRegression()


# In[355]:


logi_model.fit(x_train,y_train)


# In[356]:


pred=logi_model.predict(x_test)
pred


# In[357]:


accuracy_score(pred,y_test)


# In[358]:


precision_score(pred,y_test)


# In[363]:


models = {
    '        Logistic Regression : ': LogisticRegression(),
    '        Decision Tree : ': DecisionTreeClassifier(),
    '        Random Forest Classifier : ': RandomForestClassifier(),
}


accuracy, precision, recall = {}, {}, {}

for i in models.keys():
    
    models[i].fit(x_train, y_train)
    y_pred = models[i].predict(x_test)
    
    accuracy[i] = accuracy_score(y_pred, y_test)
    precision[i] = precision_score(y_pred, y_test)


# In[365]:


df=pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision'])
df['Accuracy'] = accuracy.values()
df['Precision'] = precision.values()
df


# In[ ]:




