import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')

url = 'beer_rating_files/model_data.csv'

df = pd.read_csv(url, low_memory= False)



df.drop('Unnamed: 0',axis=1,inplace=True)
print(df.head())
target = 'review_overall'
X = model_data.drop(target, axis = 1)
Y = model_data[target]
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state= 42)

from sklearn.preprocessing import MinMaxScaler
# Feature Scaling
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(X_train) #ONLY FIT to train data!!
scaled_test = scaler.transform(X_test)

#Preparing Logistics Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(penalty='l2', solver='sag', max_iter=10)
logmodel.fit(scaled_train,y_train)
predictions_scaled = logmodel.predict(scaled_test)

import pickle
# # Saving model to disk
pickle.dump(ET_Model, open('logmodel.pkl','wb'))
model=pickle.load(open('logmodel.pkl','rb'))

import pickle
def save_model(model_name, model):
    '''
    model_name = name.pkl
    joblib.load('name.pkl')
    assign a variable to load model
    '''
    with open(str(model_name), 'wb') as f:
        pickle.dump(model, f)