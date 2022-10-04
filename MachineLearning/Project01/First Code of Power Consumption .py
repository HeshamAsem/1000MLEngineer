import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel


dataset = pd.read_csv('household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
for col in dataset.columns:
    print(col)
# mark all missing values
nan = np.nan
dataset.replace('?', nan, inplace=True)
dataset = dataset.replace(r'^\s*$', nan, regex=True)

print(dataset.isnull().sum().sum())


for i in dataset.columns: # Excluding date and time
    dataset.loc[:, i].astype(float) # Defining the format

values = dataset.values.astype(float) # create object for dataframe values

dataset['General_Power'] = np.sqrt(np.square(values[:,0]) + np.square(values[:,1]))


# Make imputting for data that change all NaN values to most frequent 
ImputedModule = SimpleImputer(missing_values= nan, strategy ='most_frequent')
ImputedX = ImputedModule.fit(dataset)
X = ImputedX.transform(dataset)

# df = pd.DataFrame(X[1:,1:])
df = pd.DataFrame(X)

print(df.isnull().sum().sum())

df[:5]

#X Data
X = df.iloc[:,:-1]

#y Data
y = df.iloc[:,-1]


#Feature Selection 

# LinearRegressionModel = LinearRegression()

# FeatureSelection = SelectFromModel(estimator = LinearRegressionModel , max_features = None) # make sure that this model is well-defined
# X = FeatureSelection.fit_transform(X, y)

# print('Selected Features are : ' , X.get_support())

# Data Scaling

X = MinMaxScaler().fit_transform(X)
# y = MinMaxScaler().fit_transform(y.values.reshape(-1,1))
y = MinMaxScaler().fit_transform(y)

#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)


# Applying LinearRegression Model

LinearRegressionModel = LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1)
LinearRegressionModel.fit(X_train, y_train)

LinearRegressionModel.score(X_train, y_train)
LinearRegressionModel.score(X_test, y_test)

y_pred = LinearRegressionModel.predict(X_test)

y_test[:10]
y_pred[:10]

#Applying SVR Model 

# '''
# sklearn.svm.SVR(kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, tol=0.001,
#                 C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False,max_iter=-1)
# '''

# SVRModel = SVR(C = 1.0 ,epsilon=0.1,kernel = 'rbf') 

# SVRModel.fit(X_train, y_train)

# #Calculating Details
# print('SVRModel Train Score is : ' , SVRModel.score(X_train, y_train))
# print('SVRModel Test Score is : ' , SVRModel.score(X_test, y_test))
# print('----------------------------------------------------')

# #Calculating Prediction
# y_pred = SVRModel.predict(X_test)
# print('Predicted Value for SVRModel is : ' , y_pred[:10])

#----------------------------------------------------
#Calculating Mean Absolute Error
MAEValue = metrics.mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue)

#----------------------------------------------------
#Calculating Mean Squared Error
MSEValue = metrics.mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)

#----------------------------------------------------
#Calculating Median Squared Error
MdSEValue = metrics.median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue )


plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, LinearRegressionModel.predict(X_train), color = 'blue')
# plt.plot(X_train, SVRModel.predict(X_train), color = 'blue')
plt.title('SAT degrees')
plt.xlabel('Power Consumption')
plt.ylabel('General Power')
plt.show()
