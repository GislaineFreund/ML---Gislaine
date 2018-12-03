# -*- coding: utf-8 -*-
"""

@author: Gislaine
"""

import pandas as pd
import numpy as np

registro = pd.read_csv('registros3.csv', sep=';')

registro['DATA_HORA'] = pd.to_datetime(registro['DATA_HORA'])


# Gerando novas features
registro['DATA'] = registro['DATA_HORA'].dt.strftime('%d/%m/%y')
registro['HORA'] = registro['DATA_HORA'].dt.strftime('%H').astype(int)
registro['PERIODO'] = np.around(registro['HORA']/6).astype(int)

registro['DATA'] = pd.to_datetime(registro['DATA'])
registro['DIADOANO'] = registro['DATA'].apply(lambda x: x.diadoano)


# Fazendo a Limpeza nos dados
X= registro.drop(['TEMPO_SOLUCAO'],axis=1)
X= X.drop(['DATA'],axis=1)
X= X.drop(['DATA_HORA'],axis=1)
X= X.drop(['HORA'],axis=1)
y= registro['TEMPO_SOLUCAO']


#Tirando os dados em branco e colocando Zero
X = np.nan_to_num(X) 
y = np.nan_to_num(y)

#Fazendo Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Teste com Regressão Linear
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

y_predito = model.predict(X_test)
y_predito = np.around(y_predito/60).astype(int) #Em todos os casos converto minutos em horas e depois arredondo para normalizar

from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, f1_score

print (r2_score(np.around(y_test/60).astype(int), y_predito))
print(accuracy_score(np.around(y_test/60).astype(int), y_predito))
print(mean_squared_error(np.around(y_test/60).astype(int), y_predito))
print(f1_score(np.around(y_test/60).astype(int), y_predito, average='macro'))

#Teste com RN MLPR
from sklearn.neural_network import MLPRegressor
modelmlpR = MLPRegressor(alpha=0.01)
modelmlpR.fit(X_train,y_train)

y_preditoMlpr = modelmlpR.predict(X_test)
y_preditoMlpr = np.around(y_preditoMlpr/60).astype(int)

from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, f1_score
print(r2_score(np.around(y_test/60).astype(int), y_preditoMlpr))
print(accuracy_score(np.around(y_test/60).astype(int), y_preditoMlpr))
print(mean_squared_error (np.around(y_test/60).astype(int), y_preditoMlpr))
print(f1_score(np.around(y_test/60).astype(int), y_preditoMlpr,average='macro'))

#Teste com SVR
from sklearn.svm import SVR
svRmodel = SVR()
svRmodel.fit(X_train,y_train)

y_preditoSVR = svRmodel.predict(X_test)
y_preditoSVR = np.around(y_preditoSVR/60).astype(int)

from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, f1_score
print(r2_score(np.around(y_test/60).astype(int), y_preditoSVR))
print(accuracy_score(np.around(y_test/60).astype(int), y_preditoSVR))
print(mean_squared_error(np.around(y_test/60).astype(int), y_preditoSVR))
print(f1_score(np.around(y_test/60).astype(int), y_preditoSVR, average='macro'))

#Teste com Ridge
from sklearn.linear_model import Ridge
modelRidge = Ridge(alpha=0.001)
modelRidge.fit(X_train,y_train)

y_preditoRidge = modelRidge.predict(X_test)

y_preditoRidge = np.around(y_preditoRidge/60).astype(int)

from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, f1_score
print(r2_score(np.around(y_test/60).astype(int), y_preditoRidge))
print(accuracy_score(np.around(y_test/60).astype(int), y_preditoRidge))
print(mean_squared_error(np.around(y_test/60).astype(int), y_preditoRidge))
print(f1_score(np.around(y_test/60).astype(int), y_preditoRidge, average='macro'))

#Teste com Lasso
from sklearn.linear_model import Lasso
modellasso = Lasso(alpha=0.01)
modellasso.fit(X_train,y_train)

y_preditolasso = modellasso.predict(X_test)

y_preditolasso = np.around(y_preditolasso/60).astype(int)

from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, f1_score
print(r2_score(np.around(y_test/60).astype(int), y_preditolasso))
print(accuracy_score(np.around(y_test/60).astype(int), y_preditolasso))
print(mean_squared_error(np.around(y_test/60).astype(int), y_preditolasso))
print(f1_score(np.around(y_test/60).astype(int), y_preditolasso, average='macro'))

#Teste com RandomForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

regr = RandomForestRegressor(max_depth=8, random_state=0,n_estimators=100)
regr.fit(X_train, y_train)

y_predict_rf = regr.predict(X_test)

y_predict_rf = np.around(y_predict_rf/60).astype(int)

from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, f1_score
print(r2_score(np.around(y_test/60).astype(int), y_predict_rf))
print(accuracy_score(np.around(y_test/60).astype(int), y_predict_rf))
print(mean_squared_error(np.around(y_test/60).astype(int), y_predict_rf))
print(f1_score(np.around(y_test/60).astype(int), y_predict_rf, average='macro'))
