# Default imports

import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

# Write your solution here:
def percentile_k_features (df,k=20):
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    fs = SelectPercentile(f_regression, percentile=k)
    fs.fit_transform(X, y)
    support = fs.get_support()
    full_set = X.columns.values
    temp = full_set[support]
    finallist = temp.tolist()
    return finallist
