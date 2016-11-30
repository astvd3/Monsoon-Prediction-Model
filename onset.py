import pandas as pd
import numpy as np
from sklearn import neural_network
from sklearn import preprocessing
df_ip=pd.read_csv('onset_ip.csv')
df_op=pd.read_csv('onset_op.csv')

print(df_ip.T.head())
print(df_op.head())
print(len(df_op))
print(len(df_ip))
print(len(df_ip.T))
ip_frame=[]
print(len(df_ip.T[1:37]))
for i in range(0,43):
    temp=np.array(df_ip.T[(i)*12:i*12+36].values)
    temp=temp.ravel()
    ip_frame.append(temp)

print(len(ip_frame[1:30]))
print(len(ip_frame))
#print(ip_frame)
df_op_a=np.asarray(df_op)
#print(df_op_a)

df_op_ap=preprocessing.scale(df_op_a)
print(df_op_ap)
clf=neural_network.MLPRegressor(random_state=0,hidden_layer_sizes=500,activation='tanh',max_iter=500)
clf.fit(ip_frame[1:30],df_op_a[1:30])
print(clf.score(ip_frame[1:30],df_op_a[1:30]))


from sklearn import ensemble

clf2=ensemble.AdaBoostRegressor(random_state=0)
clf2.fit(ip_frame[1:30],df_op_a[1:30])

print(clf2.score(ip_frame[1:30],df_op_a[1:30]))
pred_train=clf2.predict(ip_frame[1:30])
pred=clf2.predict(ip_frame[30:40])
print(pred)
print(df_op_a[30:-1])
#pred_train=clf2.predict(input.T[0:384])
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(range(0,len(ip_frame[30:40])),pred,'red',range(0,len(ip_frame[30:40])),df_op_a[30:40],'blue')
plt.figure(2)
plt.plot(range(0,len(df_op_a[1:30])),pred_train,'red',range(0,len(df_op_a[1:30])),df_op_a[1:30],'blue')
plt.show()