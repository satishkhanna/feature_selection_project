# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def rf_rfe (df):
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    model = RandomForestClassifier()
    rfe = RFE(model, n_features_to_select=17)
    rfe.fit(X,y)
    featurelist = rfe.support_
    full_set = X.columns.values
    temp = full_set[featurelist]
    top_features = temp.tolist()
    return top_features

# Your solution code here
