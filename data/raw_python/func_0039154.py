def predict(self, X):
        """Returns the predictions for ``X``.

        Under the hood this method simply goes through the outputs of ``iter_predict`` and returns
        the final one.

        Arguments:
            X (array-like or sparse matrix of shape (n_samples, n_features)): The input samples.
                Sparse matrices are accepted only if they are supported by the weak model.

        Returns:
            array of shape (n_samples,) containing the predicted values.
        """
        return collections.deque(self.iter_predict(X), maxlen=1).pop()