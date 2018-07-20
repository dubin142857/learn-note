import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

seed = 7
test_size =0.33

# load data
data =pd.read_csv('pima-indians-diabetes.csv')
data.head(3).append(data.tail(3))

data.describe()

def splitXy(data):
    dataset =data.values
    X=dataset[:,0:-1]
    y=dataset[:,-1]

    return X,y

# split data into X and y
X,y =splitXy(data)
#split data into train and test sets
X_train, X_test,y_train, y_test = \
train_test_split(X, y, test_size=test_size,
random_state=seed)

print('The size of X is',X.shape)
print('The size of y is',y.shape)
print('The size of X_train is',X_train.shape)
print('The size of y_train is',y_train.shape)
print('The size of X_test is',X_test.shape)
print('The size of y_test is',y_test.shape)

def fit(X,y):
    model= XGBClassifier()
    model.fit(X,y)

    return model

model = fit(X_train,y_train)
print (model)

def predict(model,X,y):
    # make predictions for X
    y_pred =model.predict(X)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y,predictions)

    return predictions,accuracy

_,accuracy = predict(model,X_test,y_test)
print("Accuracy: %.2f%%" %(accuracy*100.0))




import pickle

#save model to file
pickle.dump(model,open("pima.dat","wb"))

#load model from file
pima_model = pickle.load(open("pima.dat","rb"))

_,accuracy = predict(pima_model,X_test,y_test)
print("Accuracy: %.2f%%" %(accuracy*100.0))


from graphviz import Digraph

dot = Digraph(comment='The Round Table')



from graphviz import Digraph

from  xgboost import plot_tree

plot_tree(model , num_trees = 4)

plt.show()
plt.savefig('Tree from Top to Bottom.png')
            
plot_tree(model,num_trees=0,rankdir='LR') 
plt.show()
plt.savefig('Tree from Left to Right.png')


