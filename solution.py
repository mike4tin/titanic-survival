"""Mike Fortin - Kaggle Titanic Survival Competition"""

# Use pandas to get your dataframe opened and then use .describe() to get statistics
# Choose a model from sklearn to fit the model
# Use MAE(Mean Absolute Error) to get an idea of how far off your predictions were
# train_test_split allows you to get a split of training data
# Fit a model on the train_X, train_y and then use val_X and val_y to check things out
# Use loops and values such as max_leaf_nodes to determine what the best tunable values are

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score
import pandas as pd
import numpy as np
import csv

"""Preprocessing / Setup Of Training and Validation Data"""
train_data_path = "/Users/Mike/Desktop/TitanicSurvival/train.csv"
val_data_path = "/Users/Mike/Desktop/TitanicSurvival/test.csv"
result_data_path = "/Users/Mike/Desktop/TitanicSurvival/gender_submission.csv"

val_y = pd.read_csv(result_data_path)
val_y = val_y.Survived
#print(val_y)
val_X = pd.read_csv(val_data_path)
raw_train = pd.read_csv(train_data_path)
train_y = raw_train.Survived
#print(raw_train.columns)

"""Feature Selection"""
train_X_columns = ["Pclass","Sex","Age", "SibSp", "Parch"]
train_X = raw_train[train_X_columns]
val_X = val_X[train_X_columns]

"""Encoding Male and Female To Digits and Removing NA's"""
le = LabelEncoder()
train_X["Sex"] = le.fit_transform(train_X["Sex"])
train_X = train_X.fillna(0)
#print(train_X)
val_X["Sex"] = le.fit_transform(val_X["Sex"]).copy()
val_X = val_X.fillna(0)

"""Training and Predicition"""
#Change the below value for max_leaf_nodes based on output of fine tuning
titanic_classifier = DecisionTreeRegressor(max_leaf_nodes=8, random_state=1)
titanic_classifier.fit(train_X, train_y)
output = titanic_classifier.predict(val_X)
final_output = np.where(output > .5, 1, 0)

"""Function taken from kaggle ml course """
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

"""Calculation of metrics for model validation"""
mae = mean_absolute_error(val_y, output)
accuracy = accuracy_score(val_y, final_output)

"""Fine tuning the model for optimal output"""
check_leaf_nodes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in range(1, 100):
    check_leaf_nodes.append(5 + 5 * i)
#print(check_leaf_nodes)

mae_list = []
for num in check_leaf_nodes:
    mae_list.append(get_mae(num, train_X, val_X, train_y, val_y))

"""Print Relevant Information To The Terminal"""
print("\nThe best mean absolute error is : " + str(min(mae_list)) + "\nand the corresponding value is : "
      + str(check_leaf_nodes[mae_list.index(min(mae_list))]))
print("Final test set accuracy : " + str(accuracy))
#print(final_output)

"""Write To CSV For Submission"""
# Generate passenger_id's
passenger_id_list = []
for i in range(892, 892+418):
    passenger_id_list.append(i)

with open('results.csv', 'wb') as results:
    fieldnames = ['PassengerId', 'Survived']
    result_writer = csv.DictWriter(results, fieldnames=fieldnames)
    result_writer.writeheader()
    for i in passenger_id_list:
        result_writer.writerow({'PassengerId': i, 'Survived': final_output[i-892]})



