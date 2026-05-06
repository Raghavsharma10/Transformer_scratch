def decision_function(self, X):
        "Decision function i.e. the raw data of the prediction"
        self._X = Model.convert_features(X)
        self._eval()
        return self._ind[0].hy