#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Step1: Import the necessory modules and packages 

from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

# Step2: Import the data set

x, y = make_classification(
    n_samples=100,
    n_features=1,
    n_classes=2,
    n_clusters_per_class=1,
    flip_y=0.03,
    n_informative=1,
    n_redundant=0,
    n_repeated=0
)


# Step3: Visualize the data

plt.scatter(x, y, c=y, cmap='rainbow')
plt.title('Scatter Plot of Logistic Regression')
plt.show()

# Step4: Split the data set into training and test data

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)


# Step5: Perform Logestic Regression and create the model with training data

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

# Step6: Make predections with the model using test data

y_pred = log_reg.predict(x_test)

#sStep7: Check the accuracy of the model by analyxing confusion matrix

confusion_matrix(y_test, y_pred)


# In[2]:


# Step2: Import the data set

x, y = make_classification(
    n_samples=100,
    n_features=1,
    n_classes=2,
    n_clusters_per_class=1,
    flip_y=0.03,
    n_informative=1,
    n_redundant=0,
    n_repeated=0
)


# In[3]:


print(x)
print(y)


# In[4]:


# Step3: Visualize the data

plt.scatter(x, y, c=y, cmap='rainbow')
plt.title('Scatter Plot of Logistic Regression')
plt.show()


# In[5]:


# Step4: Split the data set into training and test data

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)


# In[6]:


# Step5: Perform Logestic Regression and create the model with training data

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)


# In[7]:


# Step6: Make predections with the model using test data

y_pred = log_reg.predict(x_test)


# In[8]:


#sStep7: Check the accuracy of the model by analyxing confusion matrix

confusion_matrix(y_test, y_pred)


# In[ ]:





# In[ ]:




