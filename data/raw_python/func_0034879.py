def transform(self, X):
        """
        Discretize X using a split scheme obtained with CAIM.
        Parameters
        ----------
        X : array-like or pandas dataframe, shape [n_samples, n_features]
            Input array can contain missing values
        Returns
        -------
        X_di : sparse matrix if sparse=True else a 2-d array, dtype=int
            Transformed input.
        """

        if isinstance(X, pd.DataFrame):
            self.indx = X.index
            self.columns = X.columns
            X = X.values
        X_di = X.copy()
        categorical = self.categorical

        scheme = self.split_scheme
        for j in range(X.shape[1]):
            if j in categorical:
                continue
            sh = scheme[j]
            sh[-1] = sh[-1] + 1
            xj = X[:, j]
            # xi = xi[np.invert(np.isnan(xi))]
            for i in range(len(sh) - 1):
                ind = np.where((xj >= sh[i]) & (xj < sh[i + 1]))[0]
                X_di[ind, j] = i
        if hasattr(self, 'indx'):
            return pd.DataFrame(X_di, index=self.indx, columns=self.columns)
        return X_di