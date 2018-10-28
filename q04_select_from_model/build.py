# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(data):
    X = data.drop(['SalePrice'], axis=1)
    y = data['SalePrice']
    
    np.random.seed(9)
    
    rfc = RandomForestClassifier()
    rfc = rfc.fit(X, y)
    
    sfm = SelectFromModel(rfc, prefit=True)
    return list(X.columns[sfm.get_support()])
select_from_model(data)


