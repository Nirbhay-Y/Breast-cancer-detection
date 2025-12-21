import pandas as pd
from sklearn.feature_selection import mutual_info_classif

def select_top_features(X, y, k = 10):
    mi = mutual_info_classif(X, y, random_state = True)
    mi_Series = pd.Series(mi, index = X.columns)
    top_features = mi_Series.sort_values(ascending = False).head(k).index.to_list()
    return top_features