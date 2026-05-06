def predict_MCMC(self, X, ddof=1, full_MC=False, rejection_func=None, **kwargs):
        """Make a prediction using MCMC samples.
        
        This is essentially a convenient wrapper of :py:meth:`compute_from_MCMC`,
        designed to act more or less interchangeably with :py:meth:`predict`.
        
        Computes the mean of the GP posterior marginalized over the
        hyperparameters using iterated expectations. If `return_std` is True,
        uses the law of total variance to compute the variance of the GP
        posterior marginalized over the hyperparameters. If `return_cov` is True,
        uses the law of total covariance to compute the entire covariance of the
        GP posterior marginalized over the hyperparameters. If both `return_cov`
        and `return_std` are True, then both the covariance matrix and standard
        deviation array will be returned.
        
        Parameters
        ----------
        X : array-like (`M`,) or (`M`, `num_dim`)
            The values to evaluate the Gaussian process at.
        ddof : int, optional
            The degree of freedom correction to use when computing the variance.
            Default is 1 (standard Bessel correction for unbiased estimate).
        return_std : bool, optional
            If True, the standard deviation is also computed. Default is True.
        full_MC : bool, optional
            Set to True to compute the mean and covariance matrix using Monte
            Carlo sampling of the posterior. The samples will also be returned
            if full_output is True. Default is False (don't use full sampling).
        rejection_func : callable, optional
            Any samples where this function evaluates False will be rejected,
            where it evaluates True they will be kept. Default is None (no
            rejection). Only has an effect if `full_MC` is True.
        ddof : int, optional
        **kwargs : optional kwargs
            All additional kwargs are passed directly to
            :py:meth:`compute_from_MCMC`.
        """
        return_std = kwargs.get('return_std', True)
        return_cov = kwargs.get('return_cov', False)
        if full_MC:
            kwargs['return_mean'] = False
            kwargs['return_std'] = False
            kwargs['return_cov'] = False
            kwargs['return_samples'] = True
        else:
            kwargs['return_mean'] = True
        return_samples = kwargs.get('return_samples', True)
        res = self.compute_from_MCMC(X, **kwargs)
        
        out = {}
        
        if return_samples:
            samps = scipy.asarray(scipy.hstack(res['samp']))
        
        if full_MC:
            if rejection_func:
                good_samps = []
                for samp in samps.T:
                    if rejection_func(samp):
                        good_samps.append(samp)
                if len(good_samps) == 0:
                    raise ValueError("Did not get any good samples!")
                samps = scipy.asarray(good_samps, dtype=float).T
            mean = scipy.mean(samps, axis=1)
            cov = scipy.cov(samps, rowvar=1, ddof=ddof)
            std = scipy.sqrt(scipy.diagonal(cov))
        else:
            means = scipy.asarray(res['mean'])
            mean = scipy.mean(means, axis=0)
            
            # TODO: Allow use of robust estimators!
            if 'cov' in res:
                covs = scipy.asarray(res['cov'])
                cov = scipy.mean(covs, axis=0) + scipy.cov(means, rowvar=0, ddof=ddof)
                std = scipy.sqrt(scipy.diagonal(cov))
            elif 'std' in res:
                vars_ = scipy.asarray(scipy.asarray(res['std']))**2
                std = scipy.sqrt(scipy.mean(vars_, axis=0) +
                                 scipy.var(means, axis=0, ddof=ddof))
            if 'mean_func' in res:
                mean_funcs = scipy.asarray(res['mean_func'])
                cov_funcs = scipy.asarray(res['cov_func'])
                mean_func = scipy.mean(mean_funcs, axis=0)
                cov_func = scipy.mean(cov_funcs, axis=0) + scipy.cov(mean_funcs, rowvar=0, ddof=ddof)
                std_func = scipy.sqrt(scipy.diagonal(cov_func))
                
                mean_without_funcs = scipy.asarray(res['mean_without_func'])
                cov_without_funcs = scipy.asarray(res['cov_without_func'])
                mean_without_func = scipy.mean(mean_without_funcs, axis=0)
                cov_without_func = (
                    scipy.mean(cov_without_funcs, axis=0) +
                    scipy.cov(mean_without_funcs, rowvar=0, ddof=ddof)
                )
                std_without_func = scipy.sqrt(scipy.diagonal(cov_without_func))
                
                out['mean_func'] = mean_func
                out['cov_func'] = cov_func
                out['std_func'] = std_func
                out['mean_without_func'] = mean_without_func
                out['cov_without_func'] = cov_without_func
                out['std_without_func'] = std_without_func
        
        out['mean'] = mean
        if return_samples:
            out['samp'] = samps
        if return_std or return_cov:
            out['std'] = std
        if return_cov:
            out['cov'] = cov
        
        return out