import pandas as pd
import GWCutilities as util

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n-----\n")

#Create a variable to read the dataset
df = pd.read_csv("heartDisease_2020_sampling.csv")

print(
    "We will be performing data analysis on this Indicators of Heart Disease Dataset. Here is a sample of it: \n"
)

#Print the dataset's first five rows
print(df.head())

input("\n Press Enter to continue.\n")



#Data Cleaning
#Label encode the dataset

df = util.labelEncoder(df, ["HeartDisease", "GenHealth", "Smoking", "AlcoholDrinking", "Sex", "AgeCategory", "PhysicalActivity"])

print("\nHere is a preview of the dataset after label encoding. \n")

print(df.head())
input("\nPress Enter to continue.\n")

#One hot encode the dataset
df = util.oneHotEncoder(df, ["Race"])

print(
    "\nHere is a preview of the dataset after one hot encoding. This will be the dataset used for data analysis: \n"
)
print(df.head())


input("\nPress Enter to continue.\n")



#Creates and trains Decision Tree Model
from sklearn.model_selection import train_test_split

X = df.drop("HeartDisease", axis=1)
Y = df["HeartDisease"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 5)
print(X_train.head())

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth = 10, class_weight = "balanced")
clf = clf.fit(X_train, Y_train)




#Test the model with the testing data set and prints accuracy score
test_predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score
test_acc = accuracy_score(Y_test, test_predictions)

print("\nThe accuracy score of the training Decision Tree model is: " +  str(test_acc))



#Prints the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, test_predictions, labels = [1,0])

print("\nThe confusion matrix of the Decision Tree model is: ")

print(str(cm))


#Test the model with the training data set and prints accuracy score

train_predictions = clf.predict(X_train)

from sklearn.metrics import accuracy_score
train_acc = accuracy_score(Y_train, train_predictions)

print("\nThe accuracy score of the training Decision Tree model is: " +  str(train_acc))


input("\nPress Enter to continue.\n")



#Prints another application of Decision Trees and considerations
print("\nBelow is another application of decision trees and considerations for using them:\n")


print("A decision tree can be used in meterology to predict the weather. For example, a decision tree can be used to predict whether it will rain or not based on the temperature, humidity, and wind speed. The decision tree can be trained on historical weather data and then used to predict the weather for a certain day.")



#Prints a text representation of the Decision Tree
print("\nBelow is a text representation of how the Decision Tree makes choices:\n")
input("\nPress Enter to continue.\n")

util.printTree(clf, X.columns)

