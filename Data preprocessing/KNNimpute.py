import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

df_lab = pd.read_csv("testjoin_lab_med.csv")
print(df_lab.head())

lab_value = df_lab.iloc[:, 3:10].values
#print(lab_value)

imputer = KNNImputer(n_neighbors=10)
imputed_lab_value = imputer.fit_transform(lab_value)
#print(imputed_lab_value)

df_lab['dbp'] = imputed_lab_value[:,0]
df_lab['sbp'] = imputed_lab_value[:,1]
df_lab['blood_glucose'] = imputed_lab_value[:,2]
df_lab['hr'] = imputed_lab_value[:,3]
df_lab['PH'] = imputed_lab_value[:,4]
df_lab['bos'] = imputed_lab_value[:,5]
df_lab['temp'] = imputed_lab_value[:,6]

print(df_lab.head())

df_lab.to_csv("testjoin_lab_med_imputed.csv", sep='\t')