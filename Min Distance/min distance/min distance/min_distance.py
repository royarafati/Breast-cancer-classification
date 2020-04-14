import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#Loading Data :
df= pd.read_csv (r'E:\Arshad\pattern recognition\Poject\Classification\KNN\CancerData\CancerData\data cancer.xls')

#droping out id column as it has not any rol:
df.drop(['Unnamed: 32', 'id'],inplace=True, axis = 1)

# Missing Values column32:
df.dropna(axis=1,how='all',inplace=True)

#Converting the diagnosis value of M and B to a numerical value M (Malignant) = 1 & B (Benign) = 0
def diagnosis_value(diagnosis): 
    if diagnosis == 'M': 
        return 1
    else: 
        return 0
  
df['diagnosis'] = df['diagnosis'].apply(diagnosis_value)


# Input and Output data (data and target):
X = np.array(df.iloc[:, 1:]) 
y = np.array(df['diagnosis'])

#Splitting data to training and testing:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

#Normalizing Data :
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#fit the model into MinDistance:
from sklearn.neighbors import NearestCentroid
clf = NearestCentroid()
clf.fit(X_train, y_train)


#Prediction Score:
clf.score(X_test, y_test)
y_pred = clf.predict(X_test)


# perform 10 fold cross validation 
scores = cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'accuracy') 
print("Accuracy: %0.2f" % (scores.mean()))


#Confusion Matrix:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#Accuracy:
print("Confusion Matrix Accuracy: %0.2f" % (accuracy_score(y_test, y_pred)))
