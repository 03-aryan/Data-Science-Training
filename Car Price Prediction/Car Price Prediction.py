#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn


# In[2]:


cardf = pd.read_csv("CarPrice.csv")


# In[3]:


cardf


# In[8]:


cardf.head()


# In[4]:


cardf.isnull().sum()


# In[5]:


cardf.info


# In[6]:


cardf.describe()


# In[9]:


cardf.shape


# In[11]:


cardf.head(150)


# In[10]:


cardf.columns


# In[12]:


df = cardf[[ 'fueltype',
         'wheelbase',
       'carlength', 'carwidth', 'carheight', 'curbweight', 'enginetype',
       'cylindernumber', 'enginesize', 'fuelsystem', 'boreratio', 'stroke',
       'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg',
       'price']]


# In[13]:


df.head()


# In[14]:


df = pd.get_dummies(df,drop_first = True)


# In[15]:


df.head()


# In[20]:


corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20,20))
g = sns.heatmap(df[top_corr_features].corr(),annot = True,cmap = "RdYlGn")


# In[21]:


df.columns


# In[24]:


X = df[['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
       'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'fueltype_gas']]


# In[28]:


y = df['price']


# In[29]:


X.shape


# In[30]:


y.shape


# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[157]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=3)


# In[158]:


scaler = StandardScaler()


# In[159]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[160]:


model = LinearRegression()


# In[161]:


model.fit(X_train,y_train)


# In[162]:


pred = model.predict(X_test)


# In[163]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[164]:



print("R2 score: ", (metrics.r2_score(pred, y_test)))


# In[56]:


sns.regplot(x=pred, y=y_test)
plt.xlabel("Predicted Price")
plt.ylabel('Actual Price')
plt.title("ACtual vs predicted price")
plt.show()


# In[58]:


sns.distplot(y_test)


# In[ ]:




