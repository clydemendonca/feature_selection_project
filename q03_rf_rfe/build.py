# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Your solution code here
def rf_rfe(data):
    X = data.drop(['SalePrice'], axis=1)
    y = data['SalePrice']

    n_features_to_select = int(len(X.columns) / 2)
    
    rf_classifier = RandomForestClassifier()
    rfe = RFE(rf_classifier, n_features_to_select=n_features_to_select)

    rfe = rfe.fit(X, y)
    
    return list(X.columns[rfe.support_])
    







