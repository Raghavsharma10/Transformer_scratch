def plot(self, X=None, n=0, ax=None, envelopes=[1, 3], base_alpha=0.375,
             return_prediction=False, return_std=True, full_output=False,
             plot_kwargs={}, **kwargs):
        """Plots the Gaussian process using the current hyperparameters. Only for num_dim <= 2.
        
        Parameters
        ----------
        X : array-like (`M`,) or (`M`, `num_dim`), optional
            The values to evaluate the Gaussian process at. If None, then 100
            points between the minimum and maximum of the data's X are used for
            a univariate Gaussian process and a 50x50 grid is used for a
            bivariate Gaussian process. Default is None (use 100 points between
            min and max).
        n : int or list, optional
            The order of derivative to compute. For num_dim=1, this must be an
            int. For num_dim=2, this must be a list of ints of length 2.
            Default is 0 (don't take derivative).
        ax : axis instance, optional
            Axis to plot the result on. If no axis is passed, one is created.
            If the string 'gca' is passed, the current axis (from plt.gca())
            is used. If X_dim = 2, the axis must be 3d.
        envelopes: list of float, optional
            +/-n*sigma envelopes to plot. Default is [1, 3].
        base_alpha : float, optional
            Alpha value to use for +/-1*sigma envelope. All other envelopes `env`
            are drawn with `base_alpha`/`env`. Default is 0.375.
        return_prediction : bool, optional
            If True, the predicted values are also returned. Default is False.
        return_std : bool, optional
            If True, the standard deviation is computed and returned along with
            the mean when `return_prediction` is True. Default is True.
        full_output : bool, optional
            Set to True to return the full outputs in a dictionary with keys:
            
                ==== ==========================================================================
                mean mean of GP at requested points
                std  standard deviation of GP at requested points
                cov  covariance matrix for values of GP at requested points
                samp random samples of GP at requested points (only if `return_sample` is True)
                ==== ==========================================================================
            
        plot_kwargs : dict, optional
            The entries in this dictionary are passed as kwargs to the plotting
            command used to plot the mean. Use this to, for instance, change the
            color, line width and line style.
        **kwargs : extra arguments for predict, optional
            Extra arguments that are passed to :py:meth:`predict`.
        
        Returns
        -------
        ax : axis instance
            The axis instance used.
        mean : :py:class:`Array`, (`M`,)
            Predicted GP mean. Only returned if `return_prediction` is True and `full_output` is False.
        std : :py:class:`Array`, (`M`,)
            Predicted standard deviation, only returned if `return_prediction` and `return_std` are True and `full_output` is False.
        full_output : dict
            Dictionary with fields for mean, std, cov and possibly random samples. Only returned if `return_prediction` and `full_output` are True.
        """
        if self.num_dim > 2:
            raise ValueError("Plotting is not supported for num_dim > 2!")
        
        if self.num_dim == 1:
            if X is None:
                X = scipy.linspace(self.X.min(), self.X.max(), 100)
        elif self.num_dim == 2:
            if X is None:
                x1 = scipy.linspace(self.X[:, 0].min(), self.X[:, 0].max(), 50)
                x2 = scipy.linspace(self.X[:, 1].min(), self.X[:, 1].max(), 50)
                X1, X2 = scipy.meshgrid(x1, x2)
                X1 = X1.flatten()
                X2 = X2.flatten()
                X = scipy.hstack((scipy.atleast_2d(X1).T, scipy.atleast_2d(X2).T))
            else:
                X1 = scipy.asarray(X[:, 0]).flatten()
                X2 = scipy.asarray(X[:, 1]).flatten()
        
        if envelopes or (return_prediction and (return_std or full_output)):
            out = self.predict(X, n=n, full_output=True, **kwargs)
            mean = out['mean']
            std = out['std']
        else:
            mean = self.predict(X, n=n, return_std=False, **kwargs)
            std = None
        
        if self.num_dim == 1:
            univariate_envelope_plot(
                X,
                mean,
                std,
                ax=ax,
                base_alpha=base_alpha,
                envelopes=envelopes,
                **plot_kwargs
            )
        elif self.num_dim == 2:
            if ax is None:
                f = plt.figure()
                ax = f.add_subplot(111, projection='3d')
            elif ax == 'gca':
                ax = plt.gca()
            if 'linewidths' not in kwargs:
                kwargs['linewidths'] = 0
            s = ax.plot_trisurf(X1, X2, mean, **plot_kwargs)
            for i in envelopes:
                kwargs.pop('alpha', base_alpha)
                ax.plot_trisurf(X1, X2, mean - std, alpha=base_alpha / i, **kwargs)
                ax.plot_trisurf(X1, X2, mean + std, alpha=base_alpha / i, **kwargs)
        
        if return_prediction:
            if full_output:
                return (ax, out)
            elif return_std:
                return (ax, out['mean'], out['std'])
            else:
                return (ax, out['mean'])
        else:
            return ax