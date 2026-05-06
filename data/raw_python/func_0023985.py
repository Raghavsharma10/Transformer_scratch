def remove_outliers(self, thresh=3, **predict_kwargs):
        """Remove outliers from the GP with very simplistic outlier detection.
        
        Removes points that are more than `thresh` * `err_y` away from the GP
        mean. Note that this is only very rough in that it ignores the
        uncertainty in the GP mean at any given point. But you should only be
        using this as a rough way of removing bad channels, anyways!
        
        Returns the values that were removed and a boolean array indicating
        where the removed points were.
        
        Parameters
        ----------
        thresh : float, optional
            The threshold as a multiplier times `err_y`. Default is 3 (i.e.,
            throw away all 3-sigma points).
        **predict_kwargs : optional kwargs
            All additional kwargs are passed to :py:meth:`predict`. You can, for
            instance, use this to make it use MCMC to evaluate the mean. (If you
            don't use MCMC, then the current value of the hyperparameters is
            used.)
        
        Returns
        -------
        X_bad : array
            Input values of the bad points.
        y_bad : array
            Bad values.
        err_y_bad : array
            Uncertainties on the bad values.
        n_bad : array
            Derivative order of the bad values.
        bad_idxs : array
            Array of booleans with the original shape of X with True wherever a
            point was taken to be bad and subsequently removed.
        T_bad : array
            Transformation matrix of returned points. Only returned if
            :py:attr:`T` is not None for the instance.
        """
        mean = self.predict(
            self.X, n=self.n, noise=False, return_std=False,
            output_transform=self.T, **predict_kwargs
        )
        deltas = scipy.absolute(mean - self.y) / self.err_y
        deltas[self.err_y == 0] = 0
        bad_idxs = (deltas >= thresh)
        good_idxs = ~bad_idxs
        
        # Pull out the old values so they can be returned:
        y_bad = self.y[bad_idxs]
        err_y_bad = self.err_y[bad_idxs]
        if self.T is not None:
            T_bad = self.T[bad_idxs, :]
            non_zero_cols = (T_bad != 0).all(axis=0)
            T_bad = T_bad[:, non_zero_cols]
            X_bad = self.X[non_zero_cols, :]
            n_bad = self.n[non_zero_cols, :]
        else:
            X_bad = self.X[bad_idxs, :]
            n_bad = self.n[bad_idxs, :]
        
        # Delete the offending points:
        if self.T is None:
            self.X = self.X[good_idxs, :]
            self.n = self.n[good_idxs, :]
        else:
            self.T = self.T[good_idxs, :]
            non_zero_cols = (self.T != 0).all(axis=0)
            self.T = self.T[:, non_zero_cols]
            self.X = self.X[non_zero_cols, :]
            self.n = self.n[non_zero_cols, :]
        self.y = self.y[good_idxs]
        self.err_y = self.err_y[good_idxs]
        self.K_up_to_date = False
        
        if self.T is None:
            return (X_bad, y_bad, err_y_bad, n_bad, bad_idxs)
        else:
            return (X_bad, y_bad, err_y_bad, n_bad, bad_idxs, T_bad)