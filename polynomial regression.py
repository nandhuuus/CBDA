#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[6]:


df = pd.read_csv("D:\LR dataset.csv")


# In[7]:


df


# In[8]:


x = df["Experinces"].values
y = df["Salary"].values


# In[9]:


x = x.reshape(-1,1)


# In[26]:


poly = PolynomialFeatures(degree=4)


# In[27]:


x_poly = poly.fit_transform(x)


# ## poly.fit(x_poly,y)

# In[28]:


lr = LinearRegression()


# In[29]:


lr.fit(x_poly,y)


# In[30]:


y_pred = lr.predict(x_poly)


# In[31]:


plt.scatter(x,y, color='blue')
plt.plot(x,y_pred, color='red')


# In[ ]:




