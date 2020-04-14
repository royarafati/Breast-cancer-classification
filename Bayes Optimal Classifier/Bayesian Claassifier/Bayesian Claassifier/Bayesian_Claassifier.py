import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
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

#PCA :
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train)
pca.n_components_
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
print("PCA components are :" )
print(pca.components_)

finalDf = pd.concat([principalDf, df[['diagnosis']]], axis = 1)

#fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1) 
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('2 component PCA', fontsize = 20)
#targets = [0,1]
#colors = ['r', 'g']
#for target, color in zip(targets,colors):
 #   indicesToKeep = finalDf['diagnosis'] == target
  #  ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
   #            , finalDf.loc[indicesToKeep, 'principal component 2']
    #           , c = color
     #          , s = 50)
#ax.legend(targets)
#ax.grid()

print("PCA Variance ratio is : ")
print(pca.explained_variance_ratio_) #Total info of data(sum)

#fiting the Data into Model(Bayesian):
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Prediction Phase :
y_pred = classifier.predict(X_test)
classifier.score(X_test, y_test)

# perform 10 fold cross validation 
 
scores = cross_val_score(classifier, X_train, y_train, cv = 10, scoring = 'accuracy') 
print("score: %0.2f" % (scores.mean()))


#Confusion Matrix:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Accuracy:
print("Confusion Matrix Accuracy is ")
print(accuracy_score(y_test, y_pred))

#Plotting ROC Curve :
from sklearn.metrics import roc_curve
from sklearn import metrics
ns_probs = [0 for _ in range(len(y_test))] # generate a no skill prediction (majority class)
probs = classifier.predict_proba(X_test) #find the probability
data_probs = probs[:, 1] #positive class only
ns_fpr, ns_tpr, thresholds = metrics.roc_curve(y_test, ns_probs)
data_fpr, data_tpr, data_thresholds = metrics.roc_curve(y_test, data_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(data_fpr, data_tpr, marker='.', label='Bayes')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
