# Default imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np


data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()

def forward_selected (df,model):
    a= []
    b=[]
    np.random.seed(6)
    features = df.iloc[:,:-1]
    target = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.3)
    #headers = list(X_train)
    headers = ['OverallQual', 'GrLivArea','BsmtFinSF1', 'GarageCars', 'KitchenAbvGr', '1stFlrSF',
                        'YearRemodAdd',
                        'LotArea', 'MasVnrArea', 'WoodDeckSF']
    #headers = ['OverallQual', 'GrLivArea','BsmtFinSF1','GarageCars','KitchenAbvGr','1stFlrSF','YearRemodAdd','LotArea','MasVnrArea','WoodDeckSF','TotalBsmtSF','TotRmsAbvGrd']
    for i in headers:
        a.append (i)
        model.fit(X_train[a],y_train)
        y_pred = model.predict(X_test[a])
        acc = r2_score(y_test,y_pred)
        if not b:
            b.append (acc)
        elif acc > b[-1]:
            b.append(acc)
        else:
            a.remove(i)
    return a,b

# Your solution code here
