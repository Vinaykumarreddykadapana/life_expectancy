#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd 
import numpy as np


# In[36]:


df=pd.read_csv(r"C:\Users\HP\Downloads\archive (36)\life_expectancy.csv")


# In[37]:


df.info()


# In[38]:


df.isna().sum()


# In[39]:


df.duplicated().sum()


# In[40]:


df.drop_duplicates(inplace=True)


# In[41]:


df.duplicated().sum()


# In[42]:


df=df.dropna()


# In[43]:


df.isna().sum()


# In[44]:


df.describe()


# In[45]:


df.corr()


# In[46]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[47]:


plt.figure(figsize=(20, 10))  # Adjust the figure size if needed
sns.boxplot(data=df)
plt.title("Boxplots for All Columns")
plt.show()


# In[48]:


df.info()


# In[64]:


unique_values = df['Country'].unique()


# In[69]:


unique_values


# In[70]:


len(unique_values)


# In[82]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the specified column
df['Country'] = label_encoder.fit_transform(df['Country'])
df['Status'] = label_encoder.fit_transform(df['Status'])


# In[83]:


df.info()


# In[84]:


X = df.drop('Life expectancy', axis=1)
y = df['Life expectancy']

print('Shape of X = ', X.shape)
print('Shape of y = ', y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)
 
print('Shape of X_train = ', X_train.shape)
print('Shape of y_train = ', y_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of y_test = ', y_test.shape)


# In[85]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


# In[86]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
 
lr.fit(X_train, y_train)
 
lr.coef_
 
lr.intercept_


# In[87]:


X_test[0, :]
 
lr.predict([X_test[0, :]])
 
lr.predict(X_test)
 
y_test
 
lr.score(X_test, y_test,)
 
y_pred = lr.predict(X_test)


# In[88]:


from sklearn.metrics import r2_score
 
r2_score(y_test, y_pred)


# In[89]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
 
poly_reg = PolynomialFeatures(degree=2)
poly_reg.fit(X_train)
X_train_poly = poly_reg.transform(X_train)
X_test_poly = poly_reg.transform(X_test)
 
X_train_poly.shape, X_test_poly.shape
 
lr = LinearRegression()
 
lr.fit(X_train_poly, y_train)
 
lr.score(X_test_poly, y_test,)
 
lr.predict([X_test_poly[0,:]])
 
y_pred = lr.predict(X_test_poly)
y_pred
 
y_test


# In[90]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
 
mse, rmse


# In[91]:


r2_score(y_test, y_pred)


# In[92]:


from sklearn.tree import DecisionTreeRegressor
 
regressor = DecisionTreeRegressor(criterion='mse')
regressor.fit(X_train, y_train)
 
regressor.score(X_test, y_test)

