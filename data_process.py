import pandas as pd
import numpy as np
from sklearn import preprocessing
df_temp=pd.read_pickle('dipole.pkl')
temp=(df_temp['0.222633'])

tempB=(temp[22:538])
#print(len(tempB))
df_temp=pd.read_pickle('la_nina.pkl')
temp=(df_temp.drop('1950',axis=1))
#print(temp)
temp=temp[9:52].as_matrix()
temp=np.array(temp)
tempA=temp.ravel()
#print(len(tempA))

df_temp=pd.read_pickle('temperature.pkl')
temp=(df_temp[707:1223])
tempC=temp['16.069447']
#print(len(tempC))

df_temp = pd.read_csv('data/state-wise-rainfall.csv')
#df_temp=pd.read_pickle('normal.pkl')
#print(df_temp.columns)
temp_s=df_temp['States']
temp_s=(list(temp_s.drop_duplicates()))
#print(df_temp[0:5])
state=[]
for i in range(1,23):
    temp23=df_temp[(i-1)*43:43*i]
    temp23=temp23.drop(['States'],axis=1)
    temp23 = temp23.drop(['District'],axis=1)
    temp23 = temp23.drop(['Year'],axis=1)
    temp23=np.array(temp23)
    temp23=temp23.ravel()
    state.append(temp23)
    #print(df_temp.head())
#print(len(state))
length = sum([len(arr) for arr in state])
#print(length)


input=[]
state_d=pd.DataFrame(state)
#print(state_d.columns)
tempA_d=pd.DataFrame(tempA)
tempA_d=tempA_d.T
tempB_d=pd.DataFrame(tempB.values)

tempB_d=tempB_d.T
tempC_d=pd.DataFrame(tempC.values)

tempC_d=tempC_d.T
#print(tempC_d)
frames=[state_d,tempA_d,tempB_d,tempC_d]
result=pd.concat(frames)

#print(result)

state_n=preprocessing.normalize(state,norm='l2')

state_nd=pd.DataFrame(state_n)

tempA_n=preprocessing.normalize(tempA,norm='l2')

tempA_nd=pd.DataFrame(tempA_n)
tempB_n=preprocessing.normalize(tempB,norm='l2')

tempB_nd=pd.DataFrame(tempB_n)
tempC_n=preprocessing.normalize(tempC,norm='l2')

tempC_nd=pd.DataFrame(tempC_n)
frames2=[state_nd,tempA_nd,tempB_nd,tempC_nd]
result_n=pd.concat(frames2)
#print(result_n)

df_temp = pd.read_csv('data/targets.csv')
test=np.array(df_temp.drop('YEAR',axis=1).values)
df_temp=pd.DataFrame(test.ravel()).T
#print(df_temp)

#print(result_n.T[6:12])

for i in range(0,516):
    temp123=(result_n.T[(i):i+36].values)
    temp123=np.array(temp123)
    temp123=temp123.ravel()
    input.append(temp123)
#print(len(input))
#print(len(input[480]))

#input=np.array(input)
#input=input.ravel()
input=pd.DataFrame(input[0:480])
input.to_pickle('inputs.pkl')
#print(input.head())
#print(len(input))
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