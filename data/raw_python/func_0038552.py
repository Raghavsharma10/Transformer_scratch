def fit(self, X, y):
        """Fit the model using X as training data and y as target values"""

        self._data = X
        self._classes = np.unique(y)
        self._labels = y
        self._is_fitted = True