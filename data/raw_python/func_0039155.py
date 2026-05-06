def iter_predict_proba(self, X, include_init=False):
        """Returns the predicted probabilities for ``X`` at every stage of the boosting procedure.

        Arguments:
            X (array-like or sparse matrix of shape (n_samples, n_features)): The input samples.
                Sparse matrices are accepted only if they are supported by the weak model.
            include_init (bool, default=False): If ``True`` then the prediction from
                ``init_estimator`` will also be returned.

        Returns:
            iterator of arrays of shape (n_samples, n_classes) containing the predicted
            probabilities at each stage
        """
        utils.validation.check_is_fitted(self, 'init_estimator_')
        X = utils.check_array(X, accept_sparse=['csr', 'csc'], dtype=None, force_all_finite=False)

        probas = np.empty(shape=(len(X), len(self.classes_)), dtype=np.float64)

        for y_pred in super().iter_predict(X, include_init=include_init):
            if len(self.classes_) == 2:
                probas[:, 1] = sigmoid(y_pred[:, 0])
                probas[:, 0] = 1. - probas[:, 1]
            else:
                probas[:] = softmax(y_pred)
            yield probas