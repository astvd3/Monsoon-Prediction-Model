import pandas as pd
import numpy as np
from sklearn import preprocessing


#### Data Processing is done in 'data_process.py'
#### Taking Data From the input files
temp_i=pd.read_csv('input.csv')
temp_i= temp_i.drop('Unnamed: 0',1)
print(temp_i.head())

temp_o=pd.read_csv('test_op.csv')
temp_o=temp_o.drop('Unnamed: 6',1)
print(temp_o.head())
#### Converting Data Intro Array
targets=np.asarray(temp_o[0:480])
print(len(targets))
print(len(temp_i.T))

#### Testing it with decision Tree

from sklearn import tree

clf=tree.DecisionTreeRegressor(random_state=0)
clf.fit(temp_i.T[0:384],targets[0:384])

print(clf.score(temp_i.T[384:],targets[384:]))

pred=clf.predict(temp_i.T[384:])
pred_train=clf.predict(temp_i.T[0:384])
print(len(pred_train[:,0]))

##### Implememnting a Predictor using Neural Network

from sklearn.neural_network import MLPRegressor

clf5=MLPRegressor(random_state=0,hidden_layer_sizes=500,activation='logistic',max_iter=500,)
clf5.fit(temp_i.T[0:384],targets[0:384])
pred=clf5.predict(temp_i.T[384:])
pred_train=clf5.predict(temp_i.T[0:384])

#### Plotting The Results

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(range(0,len(targets[384:])),pred[:,0],'red',range(0,len(targets[384:])),targets[384:,0],'blue')
plt.figure(2)
plt.plot(range(0,len(targets[0:384])),pred_train[:,0],'red',range(0,len(targets[0:384])),targets[0:384,0],'blue')
plt.show()
