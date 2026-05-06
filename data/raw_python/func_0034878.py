def fit(self, X, y):
        """
        Fit CAIM
        Parameters
        ----------
        X : array-like, pandas dataframe, shape [n_samples, n_feature]
            Input array can contain missing values
        y:  array-like, pandas dataframe, shape [n_samples]
            Target variable. Must be categorical.
        Returns
        -------
        self
        """

        self.split_scheme = dict()
        if isinstance(X, pd.DataFrame):
            # self.indx = X.index
            # self.columns = X.columns
            if isinstance(self._features, list):
                self.categorical = [X.columns.get_loc(label) for label in self._features]
            X = X.values
            y = y.values
        if self._features == 'auto':
            self.categorical = self.check_categorical(X, y)
        categorical = self.categorical
        print('Categorical', categorical)

        min_splits = np.unique(y).shape[0]

        for j in range(X.shape[1]):
            if j in categorical:
                continue
            xj = X[:, j]
            xj = xj[np.invert(np.isnan(xj))]
            new_index = xj.argsort()
            xj = xj[new_index]
            yj = y[new_index]
            allsplits = np.unique(xj)[1:-1].tolist()  # potential split points
            global_caim = -1
            mainscheme = [xj[0], xj[-1]]
            best_caim = 0
            k = 1
            while (k <= min_splits) or ((global_caim < best_caim) and (allsplits)):
                split_points = np.random.permutation(allsplits).tolist()
                best_scheme = None
                best_point = None
                best_caim = 0
                k = k + 1
                while split_points:
                    scheme = mainscheme[:]
                    sp = split_points.pop()
                    scheme.append(sp)
                    scheme.sort()
                    c = self.get_caim(scheme, xj, yj)
                    if c > best_caim:
                        best_caim = c
                        best_scheme = scheme
                        best_point = sp
                if (k <= min_splits) or (best_caim > global_caim):
                    mainscheme = best_scheme
                    global_caim = best_caim
                    try:
                        allsplits.remove(best_point)
                    except ValueError:
                        raise NotEnoughPoints('The feature #' + str(j) + ' does not have' +
                                              ' enough unique values for discretization!' +
                                              ' Add it to categorical list!')

            self.split_scheme[j] = mainscheme
            print('#', j, ' GLOBAL CAIM ', global_caim)
        return self