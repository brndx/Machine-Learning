import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import ensemble
from sklearn.ensemble import ExtraTreesClassifier

data = pd.read_csv('property_prices.csv',delimiter=',',na_values=["Unknown"])
data.drop(data.columns[[0]], axis=1, inplace=True)
data.dropna()
X = data

X.drop('suburb', axis=1, inplace=True)
X.drop('address', axis=1, inplace=True)
X.drop('date', axis=1, inplace=True)
X.drop('type', axis=1, inplace=True)
X.drop('method', axis=1, inplace=True)
X.drop('realestate_agent', axis=1, inplace=True)
X.drop('building_area', axis=1, inplace=True)
X.drop('council_area', axis=1, inplace=True)
X.drop('lattitude', axis=1, inplace=True)
X.drop('longtitude', axis=1, inplace=True)
X.drop('region_name', axis=1, inplace=True)

data['lowBand'], data['highBand'] = data['price_bands'].str.split('-', 1).str
data['lowBand'] = data['lowBand'].str[:-1]
data['highBand'] = data['highBand'].str[:-1]
Y = data['lowBand']
print(Y)
X.drop('price_bands', axis=1, inplace=True)

model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)