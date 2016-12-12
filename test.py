import pandas as pd
import numpy as np
from sklearn import preprocessing



#input=pd.DataFrame(input[0:480])
#input.to_pickle('inputs.pkl')
#########################
temp_i=pd.read_csv('input.csv')
temp_i= temp_i.drop('Unnamed: 0',1)
print(temp_i.head())
'''temp_o=pd.read_csv('output.csv')
temp_o=temp_o.drop('Unnamed: 0',1)
temp_o['0']=temp_o['0'].str.replace('\n','')
temp_o['0']=temp_o['0'].str.replace(']','')
temp_o['0']=temp_o['0'].str.replace('[','')
temp_o['0']=temp_o['0'].str.replace('     ','-')
temp_o['0']=temp_o['0'].str.replace('    ','-')
temp_o['0']=temp_o['0'].str.replace('   ','-')
temp_o['0']=temp_o['0'].str.replace('  ','-')
temp_o['0']=temp_o['0'].str.replace(' ','-')
#temp_o['0']=temp_o['0'].str.replace(' ','-')
temp_o['1'], temp_o['2'] = zip(*temp_o['0'].apply(lambda x: x.split('-', 1)))
#temp_o['3'],temp_o['4'] = zip(*temp_o['1'].apply(lambda x: x.split('-', 1)))
temp_o['2']=temp_o['2'].str.replace('-',',')
print(temp_o['2'].head())
temp_o= temp_o.drop('0',1)
temp_o= temp_o.drop('1',1)
temp_o.to_csv('test_op.csv')'''
temp_o=pd.read_csv('test_op.csv')
temp_o=temp_o.drop('Unnamed: 6',1)
print(temp_o.head())

targets=np.asarray(temp_o[0:480])
print(len(targets))
print(len(temp_i.T))

from sklearn import tree

clf=tree.DecisionTreeRegressor(random_state=0)
clf.fit(temp_i.T[0:384],targets[0:384])

print(clf.score(temp_i.T[384:],targets[384:]))
#from sklearn import ensemble

#clf2=ensemble.GradientBoostingRegressor(random_state=0)
#clf2.fit(temp_i.T[0:384],targets[0:384])

#print(clf2.score(temp_i.T[384:],targets[384:]))
pred=clf.predict(temp_i.T[384:])
pred_train=clf.predict(temp_i.T[0:384])

import matplotlib.pyplot as plt
#print(clf5.get_params())
plt.figure(1)
plt.plot(range(0,len(targets[384:])),pred,'red',range(0,len(targets[384:])),targets[384:],'blue')
plt.figure(2)
plt.plot(range(0,len(targets[0:384])),pred_train,'red',range(0,len(targets[0:384])),targets[0:384],'blue')
plt.show()
#print(input.head())
#print(len(input))
'''
targets=df_temp.T[36:]

targets.to_csv('targets.csv')
#print(len(targets))
output=[]
#print(targets[0:6].T)
for i in range(0,516):
    temp123=targets[(i):i+6].values
    print(temp123)
    temp123=np.array(temp123)
    output.append(temp123)
#print(len(output))
output_d=pd.DataFrame(np.array(output))
print(output_d.head())
test=np.asarray(output[0:480])
#print(len(test))
#print(test[0])
#output=pd.DataFrame(test)
#output.to_pickle('targets.pkl')
#print(output.head)
print(input.T[0])
print(test[0])
input=input.T
#test2=pd.DataFrame(test)
targets=targets[0:480]

from sklearn import tree

clf=tree.DecisionTreeRegressor(random_state=0)
clf.fit(input.T[0:384],targets[0:384])

print(clf.score(input.T[384:],targets[384:]))
#output_d.values.to_csv('output.csv')
#input.to_csv('input.csv')
from sklearn import ensemble

clf2=ensemble.GradientBoostingRegressor(random_state=0)
clf2.fit(input.T[0:384],targets[0:384])

print(clf2.score(input.T[384:],targets[384:]))
pred=clf2.predict(input.T[384:])
pred_train=clf2.predict(input.T[0:384])
from sklearn import ensemble

clf3=ensemble.AdaBoostRegressor(random_state=0)
clf3.fit(input.T[0:384],targets[0:384])

print(clf3.score(input.T[384:],targets[384:]))

from sklearn import ensemble

clf4=ensemble.RandomForestRegressor(random_state=0)
clf4.fit(input.T[0:384],targets[0:384])

print(clf4.score(input.T[384:],targets[384:]))
import matplotlib.pyplot as plt

parameters={ 'min_samples_split':[2,10], 'max_depth':[3, 15],'n_estimators':[10,50]}
from sklearn import grid_search

from sklearn.neural_network import MLPRegressor

clf5=MLPRegressor(random_state=0,hidden_layer_sizes=500,activation='logistic',max_iter=500,)
clf5.fit(input.T[0:384],targets[0:384])
pred=clf5.predict(input.T[384:])
pred_train=clf5.predict(input.T[0:384])
print(clf5.score(input.T[384:],targets[384:]))
print(clf5.get_params())
plt.figure(1)
plt.plot(range(0,len(targets[384:])),pred,'red',range(0,len(targets[384:])),targets[384:],'blue')
plt.figure(2)
plt.plot(range(0,len(targets[0:384])),pred_train,'red',range(0,len(targets[0:384])),targets[0:384],'blue')
plt.show()
from sklearn.externals import joblib
joblib.dump(clf5, 'clf5.pkl')

'''