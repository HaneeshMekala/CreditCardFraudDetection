import pandas as pd
import seaborn as sns
import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

data = pd.read_csv("CC.csv")

fraud = data.loc[data['Class']==1] 
normal = data.loc[data['Class']==0]


fraud.count()


sns.relplot(x = 'Amount', y = 'Time', hue='Class', data = data)

sns.relplot(x = 'Amount', y = 'Class', data = data)

X = data.iloc[:,:-1] 
y = data['Class'] 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.35)



clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X_train, y_train)

y_pred= np.array(clf.predict(X_test))
y = np.array(y_test)

print(confusion_matrix(y, y_pred)) 

print(accuracy_score(y, y_pred))

print(classification_report(y, y_pred))