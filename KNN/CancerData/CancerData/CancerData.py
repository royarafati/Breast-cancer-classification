import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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



#Drop highly correlated data:

corr_matrix = df.corr().abs() #creat correlaton matrix
#print("correlation matrix is :")
#print(corr_matrix)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))# Select upper triangle of correlation matrix
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]# Find index of feature columns with correlation greater than 0.95
df.drop(df[to_drop], axis=1,inplace=True)
#df.shape


# Input and Output data (data and target):
X = np.array(df.iloc[:, 1:]) 
y = np.array(df['diagnosis'])

#Splitting data to training and testing:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

#Normalizing Data :
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#outliers :
Q1= df.quantile(0.75)
Q3 = df.quantile(0.25)
IQR=Q3-Q1
df.boxplot() #removing area_mean & area Worst
plt.show()
#df_out =df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
#df_out.info()
#if((df< lowerQuantile) |(df > upperQuantile)):
#df_out = df[((df< (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
#df.drop(df[ (df['area_worst'] >(Q3 + 1.5 * IQR)) | (df['area_worst'] < (Q1 - 1.5 * IQR)) ].index , inplace=True)
#print(df.info())
#df.boxplot()
#plt.show()


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

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0,1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['diagnosis'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
print("PCA Variance ratio is : ")
print(pca.explained_variance_ratio_) #Total info of data(sum)



#fit the model into KNN:
knn = KNeighborsClassifier(n_neighbors = 6) 
knn.fit(X_train, y_train)


#Prediction Score:
knn.score(X_test, y_test)
y_pred = knn.predict(X_test)


# perform 10 fold cross validation 
for k in range(1, 51, 2): 
	knn = KNeighborsClassifier(n_neighbors = k) 
	scores = cross_val_score(knn, X_train, y_train, cv = 10, scoring = 'accuracy') 
print("KNN Score is: %0.2f" % (scores.mean()))

  

#Determinig the best k :

error = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred= knn.predict(X_test)
    error.append(np.mean(y_pred != y_test))


print('The optimal number of neighbors is % d ' % error.index(min(error)))

plt.figure(figsize=(12, 6))    
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()


#Confusion Matrix:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#Accuracy:
print("Confusion Matrix Accuracy: %0.2f" % (accuracy_score(y_test, y_pred)))

#ROC Curve :
from sklearn import metrics
ns_probs = [0 for _ in range(len(y_test))] # generate a no skill prediction (majority class)
probs = knn.predict_proba(X_test) #find the probability
data_probs = probs[:, 1] #positive class only
ns_fpr, ns_tpr, thresholds = metrics.roc_curve(y_test, ns_probs)
data_fpr, data_tpr, data_thresholds = metrics.roc_curve(y_test, data_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(data_fpr, data_tpr, marker='.', label='KNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
