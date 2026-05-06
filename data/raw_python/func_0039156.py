def iter_predict(self, X, include_init=False):
        """Returns the predicted classes for ``X`` at every stage of the boosting procedure.

        Arguments:
            X (array-like or sparse matrix of shape (n_samples, n_features)): The input samples.
                Sparse matrices are accepted only if they are supported by the weak model.
            include_init (bool, default=False): If ``True`` then the prediction from
                ``init_estimator`` will also be returned.

        Returns:
            iterator of arrays of shape (n_samples, n_classes) containing the predicted classes at
            each stage.
        """
        for probas in self.iter_predict_proba(X, include_init=include_init):
            yield self.encoder_.inverse_transform(np.argmax(probas, axis=1))