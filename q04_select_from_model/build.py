# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

np.random.seed(9)

def select_from_model (df):
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    model = RandomForestClassifier()
    sfm = SelectFromModel(model)
    sfm.fit(X,y)
    featurelist = sfm.get_support(indices=True)
    full_set = X.columns.values
    temp = full_set[featurelist]
    feature_name = temp.tolist()
    return feature_name
# Your solution code here
