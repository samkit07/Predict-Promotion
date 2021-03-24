import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('HRdata.csv')

dataset['education'].fillna(dataset['education'].mode()[0], inplace = True)
dataset['previous_year_rating'].fillna(dataset['previous_year_rating'].mean(), inplace=True)
dataset['previous_year_rating'] = dataset.previous_year_rating.astype(int)
dataset=dataset.drop('region', axis=1)
dataset.drop(['Unnamed: 0','employee_id'], inplace=True, axis=1)
X= dataset.iloc[:, :11]

def convert_to_int(word):
    wd={'Sales & Marketing':1, 'Operations':2, 'Technology':3, 'Analytics':4,
       'R&D':5, 'Procurement':6, 'Finance':7, 'HR':8, 'Legal':9}
    return wd[word]
X['department'] = X['department'].apply(lambda x : convert_to_int(x))
def conv(word):
    wd={"Master's & above":1, "Bachelor's":2, 'Below Secondary':3, 0:0}
    return wd[word]
X['education'] = X['education'].apply(lambda x : conv(x))
X.recruitment_channel.replace(['sourcing', 'other', 'referred'],[1,2,3], inplace=True)
X.gender.replace(to_replace = {'m':1, 'f':2}, inplace=True)
X=X.astype(int)
y = dataset.iloc[:, -1]
y=y.astype(int)

from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test = train_test_split (X,y,test_size=0.20)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
ada.fit(X_train,y_train)

from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

pickle.dump(rfc, open('model.pkl','wb'))
pickle.dump(ada, open('model1.pkl','wb'))
pickle.dump(clf, open('model2.pkl','wb'))
print(rfc.score(X_train,y_train))
print(ada.score(X_train,y_train))
print(clf.score(X_train,y_train))

model = pickle.load(open('model.pkl','rb'))
model1 = pickle.load(open('model1.pkl','rb'))
model2 = pickle.load(open('model2.pkl','rb'))

print(model.predict([['1', '1', '1', '1', '1', '37', '4', '10', '1', '0', '92']]))
print(model1.predict([['1', '1', '1', '1', '1', '37', '4', '10', '1', '0', '92']]))
print(model2.predict([['1', '1', '1', '1', '1', '37', '4', '10', '1', '0', '92']]))