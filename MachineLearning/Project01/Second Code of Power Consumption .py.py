##
# Import all useful libraries:
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , r2_score

## Import data
# Load the data from the text file

HPC = pd.read_csv('02 Household Power Consumption.txt',sep = ';')

HPC.head()  # Checking first 5 columns
HPC.shape  # Checking the shape (185711, 9)

## Cleaning data:
    
    
df_obj = HPC.select_dtypes(['object'])   #Making object that selects only strings

HPC[df_obj.columns] = df_obj.apply(lambda x: x.str.strip()) #lambda to strip strings

HPC = HPC.replace(dict.fromkeys(['','?'], np.nan)) #replacing missing data with nans

# Making sure that python knows the exact datatypes to be able to impute clean

for i in HPC.columns[2:]: # Excluding date and time
    HPC.loc[:, i].astype(float) # Defining the format

# Now we can check number of nans and remove them 

print(HPC.isnull().sum().sum())

HPC = HPC.drop(['Date','Time'],axis=1) # We wont use time series

imputer = SimpleImputer(strategy='mean')  # Replacing nans with mean
HPC = pd.DataFrame(imputer.fit_transform(HPC)) # Fitting

print(HPC.isnull().sum().sum()) # Make sure no more nans 

# Using polynomial features because it increased accuracy

poly_reg = PolynomialFeatures(degree = 2)
HPC_Data = poly_reg.fit_transform(HPC.iloc[:,1:])
HPC_Data = pd.DataFrame(HPC_Data)
HPC_Data.shape 
HPC_Data.head(2) 

# Now we define x , y

X = HPC_Data.iloc[:,:]
y = HPC.iloc[:,0]


# Time to scale !

Scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X_scaled = Scaler.fit_transform(X)
y_scaled = Scaler.fit_transform(y.values.reshape(-1,1)) # Because scaling needs 2d array
X_scaled = pd.DataFrame(X_scaled) 
y_scaled = pd.Series(y_scaled.reshape(-1)) # Because series needs 1d array XD

# Splitting data

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled,
                                                    test_size=0.33, random_state=44,
                                                    shuffle =True)


# We choosed linear regression model to predict the Global active power
#to predict what actual active power is used
Model = LinearRegression()
Model.fit(X_train, y_train)
y_pred = Model.predict(X_test)
CrossValidateScoreTrain = cross_val_score(Model, X_train, y_train, cv=3)
CrossValidateScoreTest = cross_val_score(Model, X_test, y_test, cv=3)

# Some predictions
Model.predict(X_test.iloc[0:5,:])
y_test.head(5)


# Showing Scores 
print(CrossValidateScoreTrain)
print('//////////////// ')
print(CrossValidateScoreTest)


MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)
AccScore = r2_score(y_test, y_pred)
print(AccScore)
