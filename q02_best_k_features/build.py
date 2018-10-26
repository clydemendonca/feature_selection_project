# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(data, k=20):
    X = data.drop(['SalePrice'], axis=1)
    y = data['SalePrice']

    f_reg = f_regression(X, y)
    f_reg /= np.max(f_reg)
    
    d = dict(zip(X.columns, f_reg[0]))
    
    df = pd.DataFrame(f_reg[0], columns=['score'], index=X.columns)
    
    size = int(np.ceil(df.shape[0] * k / 100))
    
    df = df.sort_values('score', ascending=False)[:size]
    
    return list(df.index)




