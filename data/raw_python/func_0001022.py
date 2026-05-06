def predict(self, X):
        """
        Predict values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """
        if self._genelm_regressor is None:
            raise ValueError("SimpleELMRegressor not fitted")

        return self._genelm_regressor.predict(X)