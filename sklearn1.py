import numpy
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import statsmodels.api as stats

data = pd.read_csv("cost.csv", sep=",", index_col=0)
output1 = data['Cost of Living Index'] #get COL column
input1 = data.drop(['Cost of Living Index', 'City', 'Local Purchasing Power Index'], axis=1) #get all other columns
structure = linear_model.LinearRegression()
model = structure.fit(input1, output1) #create linear model
predictions = model.predict(input1)
print(predictions[0:5])


