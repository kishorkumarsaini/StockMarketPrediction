import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



df=pd.read_csv('adult.csv')
df
# filling missing values
col_names = df.columns
for c in col_names:
    df[c] = df[c].replace("?", np.NaN)

df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

#discretisation
df.replace(['Divorced', 'Married-AF-spouse', 
              'Married-civ-spouse', 'Married-spouse-absent', 
              'Never-married','Separated','Widowed'],
             ['divorced','married','married','married',
              'not married','not married','not married'], inplace = True)

#label Encoder
category_col =['workclass', 'race', 'education','marital-status', 'occupation',
               'relationship', 'gender', 'native-country', 'income'] 
labelEncoder = preprocessing.LabelEncoder()

# creating a map of all the numerical values of each categorical labels.
mapping_dict={}
for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col]=le_name_mapping

#droping redundant columns
df=df.drop(['fnlwgt','educational-num'], axis=1)
df

X=df.values[:,0:12]
Y=df.values[:,12]

#split the dataset
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#make a classifier
classifier=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=5,min_samples_leaf=5)
classifier.fit(x_train,y_train)
#make prediction

y_predict=classifier.predict(x_test)

accuracy_score=accuracy_score(y_test,y_predict)
accuracy_score

#save our model
import pickle
pickle.dump(classifier,open("model.pkl","wb"))
