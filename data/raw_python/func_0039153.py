def iter_predict(self, X, include_init=False):
        """Returns the predictions for ``X`` at every stage of the boosting procedure.

        Args:
            X (array-like or sparse matrix of shape (n_samples, n_features): The input samples.
                Sparse matrices are accepted only if they are supported by the weak model.
            include_init (bool, default=False): If ``True`` then the prediction from
                ``init_estimator`` will also be returned.
        Returns:
            iterator of arrays of shape (n_samples,) containing the predicted values at each stage
        """
        utils.validation.check_is_fitted(self, 'init_estimator_')
        X = utils.check_array(X, accept_sparse=['csr', 'csc'], dtype=None, force_all_finite=False)

        y_pred = self.init_estimator_.predict(X)

        # The user decides if the initial prediction should be included or not
        if include_init:
            yield y_pred

        for estimators, line_searchers, cols in itertools.zip_longest(self.estimators_,
                                                                      self.line_searchers_,
                                                                      self.columns_):

            for i, (estimator, line_searcher) in enumerate(itertools.zip_longest(estimators,
                                                                                 line_searchers or [])):

                # If we used column sampling then we have to make sure the columns of X are arranged
                # in the correct order
                if cols is None:
                    direction = estimator.predict(X)
                else:
                    direction = estimator.predict(X[:, cols])

                if line_searcher:
                    direction = line_searcher.update(direction)

                y_pred[:, i] += self.learning_rate * direction

            yield y_pred