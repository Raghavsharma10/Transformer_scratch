def predict_proba(self, X):
        """Returns the predicted probabilities for ``X``.

        Arguments:
            X (array-like or sparse matrix of shape (n_samples, n_features)): The input samples.
                Sparse matrices are accepted only if they are supported by the weak model.

        Returns:
            array of shape (n_samples, n_classes) containing the predicted probabilities.
        """
        return collections.deque(self.iter_predict_proba(X), maxlen=1).pop()