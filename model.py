#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle


# In[5]:


dataset = pd.read_csv('diabetes.csv')


# In[8]:


dataset[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = dataset[[
    'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)


# In[9]:


dataset['Glucose'].fillna(dataset['Glucose'].mean(), inplace=True)


# In[10]:


dataset['BloodPressure'].fillna(dataset['BloodPressure'].mean(), inplace=True)


# In[11]:


dataset['SkinThickness'].fillna(dataset['SkinThickness'].mean(), inplace=True)


# In[12]:


dataset['Insulin'].fillna(dataset['Insulin'].mean(), inplace=True)


# In[13]:


dataset['BMI'].fillna(dataset['BMI'].mean(), inplace=True)


# In[14]:


x = dataset.iloc[:, :-1].values


# In[16]:


y = dataset.iloc[:, -1].values


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)


# In[20]:


# In[30]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[31]:


dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt.fit(x_train, y_train)

pickle.dump(dt, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
