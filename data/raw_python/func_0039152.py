def fit(self, X, y, eval_set=None):
        """Fit a gradient boosting procedure to a dataset.

        Args:
            X (array-like or sparse matrix of shape (n_samples, n_features)): The training input
                samples. Sparse matrices are accepted only if they are supported by the weak model.
            y (array-like of shape (n_samples,)): The training target values (strings or integers
                in classification, real numbers in regression).
            eval_set (tuple of length 2, optional, default=None): The evaluation set is a tuple
                ``(X_val, y_val)``. It has to respect the same conventions as ``X`` and ``y``.
        Returns:
            self
        """
        # Verify the input parameters
        base_estimator = self.base_estimator
        base_estimator_is_tree = self.base_estimator_is_tree
        if base_estimator is None:
            base_estimator = tree.DecisionTreeRegressor(max_depth=1, random_state=self.random_state)
            base_estimator_is_tree = True

        if not isinstance(base_estimator, base.RegressorMixin):
            raise ValueError('base_estimator must be a RegressorMixin')

        loss = self.loss or self._default_loss

        self.init_estimator_ = base.clone(self.init_estimator or loss.default_init_estimator)

        eval_metric = self.eval_metric or loss

        line_searcher = self.line_searcher
        if line_searcher is None and base_estimator_is_tree:
            line_searcher = loss.tree_line_searcher

        self._rng = utils.check_random_state(self.random_state)

        # At this point we assume the input data has been checked
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Instantiate some training variables
        self.estimators_ = []
        self.line_searchers_ = []
        self.columns_ = []
        self.eval_scores_ = [] if eval_set else None

        # Use init_estimator for the first fit
        self.init_estimator_.fit(X, y)
        y_pred = self.init_estimator_.predict(X)

        # We keep training weak learners until we reach n_estimators or early stopping occurs
        for _ in range(self.n_estimators):

            # If row_sampling is lower than 1 then we're doing stochastic gradient descent
            rows = None
            if self.row_sampling < 1:
                n_rows = int(X.shape[0] * self.row_sampling)
                rows = self._rng.choice(X.shape[0], n_rows, replace=False)

            # If col_sampling is lower than 1 then we only use a subset of the features
            cols = None
            if self.col_sampling < 1:
                n_cols = int(X.shape[1] * self.col_sampling)
                cols = self._rng.choice(X.shape[1], n_cols, replace=False)

            # Subset X
            X_fit = X
            if rows is not None:
                X_fit = X_fit[rows, :]
            if cols is not None:
                X_fit = X_fit[:, cols]

            # Compute the gradients of the loss for the current prediction
            gradients = loss.gradient(y, self._transform_y_pred(y_pred))

            # We will train one weak model per column in y
            estimators = []
            line_searchers = []

            for i, gradient in enumerate(gradients.T):

                # Fit a weak learner to the negative gradient of the loss function
                estimator = base.clone(base_estimator)
                estimator = estimator.fit(X_fit, -gradient if rows is None else -gradient[rows])
                estimators.append(estimator)

                # Estimate the descent direction using the weak learner
                direction = estimator.predict(X if cols is None else X[:, cols])

                # Apply line search if a line searcher has been provided
                if line_searcher:
                    ls = copy.copy(line_searcher)
                    ls = ls.fit(y[:, i], y_pred[:, i], gradient, direction)
                    line_searchers.append(ls)
                    direction = ls.update(direction)

                # Move the predictions along the estimated descent direction
                y_pred[:, i] += self.learning_rate * direction

            # Store the estimator and the step
            self.estimators_.append(estimators)
            if line_searchers:
                self.line_searchers_.append(line_searchers)
            self.columns_.append(cols)

            # We're now at the end of a round so we can evaluate the model on the validation set
            if not eval_set:
                continue
            X_val, y_val = eval_set
            self.eval_scores_.append(eval_metric(y_val, self.predict(X_val)))

            # Check for early stopping
            if self.early_stopping_rounds and len(self.eval_scores_) > self.early_stopping_rounds:
                if self.eval_scores_[-1] > self.eval_scores_[-(self.early_stopping_rounds + 1)]:
                    break

        return self