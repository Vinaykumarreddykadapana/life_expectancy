


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




from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

# Evaluation
from sklearn.metrics import r2_score, mean_squared_error
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

print("Random Forest R2 Score:", r2_rf)
print("Random Forest RMSE:", rmse_rf)





from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbr.fit(X_train, y_train)

y_pred_gbr = gbr.predict(X_test)

# Evaluation
r2_gbr = r2_score(y_test, y_pred_gbr)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mse_gbr)

print("Gradient Boosting R2 Score:", r2_gbr)
print("Gradient Boosting RMSE:", rmse_gbr)











from sklearn.svm import SVR

svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)

y_pred_svr = svr.predict(X_test)

# Evaluation
r2_svr = r2_score(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)

print("SVR R2 Score:", r2_svr)
print("SVR RMSE:", rmse_svr)




from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

# Evaluation
r2_knn = r2_score(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)

print("KNN R2 Score:", r2_knn)
print("KNN RMSE:", rmse_knn)








