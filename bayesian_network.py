
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
dataset = pd.read_csv(r'/content/data cancer.xls')

features = dataset.iloc[:, 2:32]
label = dataset.iloc[:, 1]
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
label = number.fit_transform(label)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
features = scaler.fit_transform(features)

scores = []
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
cv = KFold(n_splits=10, random_state=None, shuffle=False)
for train_index, test_index in cv.split(features):
    X_train, X_test, y_train, y_test = features[train_index], features[test_index], label[train_index], label[test_index]
    NB.fit(X_train, y_train)
    scores.append(NB.score(X_test, y_test))    
print(scores)

y_score = NB.predict(X_test)
#print(y_pred)

print('Accuracy of NaiveBayes classifier on training set: {:.2f}'
     .format(NB.score(X_train, y_train)))
print('Accuracy of NaiveBayes classifier on test set: {:.2f}'
     .format(NB.score(X_test, y_test)))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = NB.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

from sklearn.metrics import roc_curve, auc
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
fpr, tpr, _ = roc_curve(y_test, y_score)

plt.figure()
plt.plot(fpr, tpr, color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()