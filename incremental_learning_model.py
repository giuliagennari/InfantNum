# Giulia Gennari August 2020
# CLASS NEEDED TO IMPLEMENT INCREMENTAL LEARNING WITH MNE SLIDERS 

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
        
class MyBasicModel(SGDClassifier):
    def fit(self, X, y):
        X, y = shuffle(X, y)
        super().partial_fit(X, y, classes = np.unique(y))
        
        return self        