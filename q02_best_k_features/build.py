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
    h = fs.scores_
    temp1 = h[support]
    full_set = X.columns.values
    temp2 = full_set[support]
    finallist1 = temp1.tolist()
    finallist2 = temp2.tolist()
    z = [x for _,x in sorted(zip(finallist1,finallist2),reverse=True)]
    return z
