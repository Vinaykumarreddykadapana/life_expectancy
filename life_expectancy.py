


import pandas as pd 
import numpy as np




df=pd.read_csv(r"C:\Users\HP\Downloads\archive (36)\life_expectancy.csv")

df.info()


df.isna().sum()


df.duplicated().sum()



df.drop_duplicates(inplace=True)


df.duplicated().sum()


df=df.dropna()


df.isna().sum()


df.describe()


df.corr()


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
plt.show()


plt.figure(figsize=(20, 10))  
sns.boxplot(data=df)
plt.title("Boxplots for All Columns")
plt.show()


df.info()


unique_values = df['Country'].unique()


unique_values


len(unique_values)


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Country'] = label_encoder.fit_transform(df['Country'])
df['Status'] = label_encoder.fit_transform(df['Status'])


df.info()


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



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)



from sklearn.linear_model import LinearRegression
lr = LinearRegression()
 
lr.fit(X_train, y_train)
 
lr.coef_
 
lr.intercept_



X_test[0, :]
 
lr.predict([X_test[0, :]])
 
lr.predict(X_test)
 
y_test
 
lr.score(X_test, y_test,)
 
y_pred = lr.predict(X_test)



from sklearn.metrics import r2_score
 
r2_score(y_test, y_pred)




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



from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
 
mse, rmse



r2_score(y_test, y_pred)



from sklearn.tree import DecisionTreeRegressor
 
regressor = DecisionTreeRegressor(criterion='mse')
regressor.fit(X_train, y_train)
 
regressor.score(X_test, y_test)

