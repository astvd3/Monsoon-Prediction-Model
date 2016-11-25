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
print(state_d.columns)
tempA_d=pd.DataFrame(tempA)
tempA_d=tempA_d.T
tempB_d=pd.DataFrame(tempB.values)

tempB_d=tempB_d.T
tempC_d=pd.DataFrame(tempC.values)

tempC_d=tempC_d.T
print(tempC_d)
frames=[state_d,tempA_d,tempB_d,tempC_d]
result=pd.concat(frames)

print(result)

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
print(result_n)

df_temp = pd.read_csv('data/targets.csv')
test=np.array(df_temp.drop('YEAR',axis=1).values)
df_temp=pd.DataFrame(test.ravel()).T
print(df_temp)