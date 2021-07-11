from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

#Importing data frames 
test_df = pd.read_csv('./TitanicDataSet/test.csv')
train_df = pd.read_csv('./TitanicDataSet/train.csv')

#Looking at first few values for train data
print(train_df.head())

#Understanding data set 
print(train_df.describe())

# Defining features and target
y = train_df['Survived']
# Listing main features of the dataset ( Assuming the values depend on them)
features = ["Pclass", "Sex", "SibSp", "Parch"]
# Adding dummy values for categorical variables 
X = pd.get_dummies(train_df[features])
#Converting Test data using the same convention
X_test = pd.get_dummies(test_df[features])

#Creating random forest model 
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=1)
#Adding features and target to the model
model.fit(X,y)

#Prediction test data using the train data model
prediction = model.predict(X_test)

#Creating the submission file
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'] , 'Survived' : prediction})

#Saving the submission file 
submission.to_csv('my_submission.csv',index = False)

print("Submission done")

