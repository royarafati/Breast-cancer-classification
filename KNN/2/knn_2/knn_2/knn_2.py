
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#Loading Data :
df= pd.read_csv (r'E:\Projects\Python\Classification\KNN\2\knn_2\knn_2\data cancer.xls')

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




#Splitting data to training and testing:
X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'diagnosis'],
  df['diagnosis'], stratify=df['diagnosis'], random_state=66)

#Normalizing Data :
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


train_accuracy = []
test_accuracy = []

#fit the mode into KNN:
k = range(1, 50)
 
for n_neighbors in k:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    train_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.plot(k, train_accuracy, label="Train Accuracy")
plt.plot(k, test_accuracy, label="Test Accuracy")
plt.title('Breast Cancer Diagnosis k-Nearest Neighbor Accuracy')
plt.ylabel("Accuracy")
plt.xlabel("k")
plt.legend()
plt.show()

#Prediction Phase :
y_pred = knn.predict(X_test)

#Confusion Matrix:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Accuracy:
print(accuracy_score(y_test, y_pred))