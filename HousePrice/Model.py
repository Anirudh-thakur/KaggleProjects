import pandas as pd 
import numpy as np 

House_train = pd.read_csv("./DataSet/train.csv")
House_test = pd.read_csv("./DataSet/test.csv")

#Checking the data set 
print(House_train.head())
print(House_train.shape)

# Getting all the columns for which 
null_sum = House_train.isnull().sum()
print(null_sum[null_sum > 0].sort_values(ascending=False))

# Removing columns which have more than 1000 null values from test and train
House_train = House_train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
House_test = House_test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
print(House_train.shape)


#Getting dummy values for the rest
ColumnListObject = ['FireplaceQu', 'GarageType', 'GarageFinish','GarageQual','GarageCond',
                    'BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'MasVnrType']
ColumnListInteger = ['LotFrontage', 'GarageYrBlt','MasVnrArea']

for col in ColumnListObject:
    House_train[col] = House_train[col].fillna('Missing')
    House_test[col] = House_test[col].fillna('Missing')

for col in ColumnListInteger:
    mean = House_train[col].mean()
    House_train[col] = House_train[col].fillna(mean)
    mean = House_test[col].mean()
    House_test[col] = House_test[col].fillna(mean)


# Drop remaining null values
House_train = House_train.dropna()

# Adding dummy values for object column(Categorical variables)
col_object = [features for features in House_train.columns if House_train[features].dtypes == 'O']
for col in col_object:
    House_train[col] = pd.get_dummies(House_train[col])
    House_test[col] = pd.get_dummies(House_test[col])

#Selecting features and target        
y = House_train['SalePrice']
X = House_train.iloc[:,1:-1]


import xgboost as xgb
model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
model.fit(X, y)
Id = House_test.iloc[: , 0]
House_test = House_test.iloc[:, 1:]
prediction = model.predict(House_test)

submission = pd.DataFrame(
    {'Id': Id, 'SalePrice': prediction})

submission.to_csv('my_submission.csv', index=False)
