import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

data_xls = pd.read_excel('data/District_Rainfall_Normal_0.xls', 0, index_col=None)
data_xls.to_csv('data/District_Rainfall_Normal_0.csv', encoding='utf-8')
df_normal = pd.read_csv('data/District_Rainfall_Normal_0.csv', header=1)
data_xls = pd.read_excel('data/india_-_monthly_rainfall_data_-_1901_to_2002.xlsx', 0, index_col=None)
data_xls.to_csv('data/india_-_monthly_rainfall_data_-_1901_to_2002.csv', encoding='utf-8')
df_rainfall = pd.read_csv('data/india_-_monthly_rainfall_data_-_1901_to_2002.csv', header=1)
#data_xls = pd.read_excel('data/tas5_1900_2012.xls', 0, index_col=None)
#data_xls.to_csv('data/tas5_1900_2012.csv', encoding='utf-8')
df_temperature = pd.read_csv('data/tas5_1900_2012.csv', header=1)
df_la_nina = pd.read_csv('data/MEI.csv', header=1)
df_dipole = pd.read_csv('data/DMI.csv', header=1)

print(df_normal.head())
print(df_rainfall.head())
print(df_temperature.head())
print(df_la_nina.head())
print(df_dipole.head())

df_normal.to_pickle('normal.pkl')
df_temperature.to_pickle('temperature.pkl')
df_rainfall.to_pickle('rainfall.pkl')
df_la_nina.to_pickle('la_nina.pkl')
df_dipole.to_pickle('dipole.pkl')