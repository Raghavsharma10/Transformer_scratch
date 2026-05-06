def compute_from_MCMC(self, X, n=0, return_mean=True, return_std=True,
                          return_cov=False, return_samples=False,
                          return_mean_func=False, num_samples=1, noise=False,
                          samp_kwargs={}, sampler=None, flat_trace=None, burn=0,
                          thin=1, **kwargs):
        """Compute desired quantities from MCMC samples of the hyperparameter posterior.
        
        The return will be a list with a number of rows equal to the number of
        hyperparameter samples. The columns depend on the state of the boolean
        flags, but will be some subset of (mean, stddev, cov, samples), in that
        order. Samples will be the raw output of :py:meth:`draw_sample`, so you
        will need to remember to convert to an array and flatten if you want to
        work with a single sample.
        
        Parameters
        ----------
        X : array-like (`M`,) or (`M`, `num_dim`)
            The values to evaluate the Gaussian process at.
        n : non-negative int or list, optional
            The order of derivative to compute. For num_dim=1, this must be an
            int. For num_dim=2, this must be a list of ints of length 2.
            Default is 0 (don't take derivative).
        return_mean : bool, optional
            If True, the mean will be computed at each hyperparameter sample.
            Default is True (compute mean).
        return_std : bool, optional
            If True, the standard deviation will be computed at each
            hyperparameter sample. Default is True (compute stddev).
        return_cov : bool, optional
            If True, the covariance matrix will be computed at each
            hyperparameter sample. Default is True (compute stddev).
        return_samples : bool, optional
            If True, random sample(s) will be computed at each hyperparameter
            sample. Default is False (do not compute samples).
        num_samples : int, optional
            Compute this many samples if `return_sample` is True. Default is 1.
        noise : bool, optional
            If True, noise is included in the predictions and samples. Default
            is False (do not include noise).
        samp_kwargs : dict, optional
            If `return_sample` is True, the contents of this dictionary will be
            passed as kwargs to :py:meth:`draw_sample`.
        sampler : :py:class:`Sampler` instance or None, optional
            :py:class:`Sampler` instance that has already been run to the extent
            desired on the hyperparameter posterior. If None, a new sampler will
            be created with :py:meth:`sample_hyperparameter_posterior`. In this
            case, all extra kwargs will be passed on, allowing you to set the
            number of samples, etc. Default is None (create sampler).
        flat_trace : array-like (`nsamp`, `ndim`) or None, optional
            Flattened trace with samples of the free hyperparameters. If present,
            overrides `sampler`. This allows you to use a sampler other than the
            ones from :py:mod:`emcee`, or to specify arbitrary values you wish
            to evaluate the curve at. Note that this WILL be thinned and burned
            according to the following two kwargs. "Flat" refers to the fact
            that you must have combined all chains into a single one. Default is
            None (use `sampler`).
        burn : int, optional
            The number of samples to discard at the beginning of the chain.
            Default is 0.
        thin : int, optional
            Every `thin`-th sample is kept. Default is 1.
        num_proc : int, optional
            The number of processors to use for evaluation. This is used both
            when calling the sampler and when evaluating the Gaussian process.
            If None, the number of available processors will be used. If zero,
            evaluation will proceed in parallel. Default is to use all available
            processors.
        **kwargs : extra optional kwargs
            All additional kwargs are passed to
            :py:meth:`sample_hyperparameter_posterior`.
        
        Returns
        -------
        out : dict
            A dictionary having some or all of the fields 'mean', 'std', 'cov'
            and 'samp'. Each entry is a list of array-like. The length of this
            list is equal to the number of hyperparameter samples used, and the
            entries have the following shapes:
            
                ==== ====================
                mean (`M`,)
                std  (`M`,)
                cov  (`M`, `M`)
                samp (`M`, `num_samples`)
                ==== ====================
        """
        output_transform = kwargs.pop('output_transform', None)
        if flat_trace is None:
            if sampler is None:
                sampler = self.sample_hyperparameter_posterior(burn=burn, **kwargs)
                # If we create the sampler, we need to make sure we clean up its pool:
                try:
                    sampler.pool.close()
                except AttributeError:
                    # This will occur if only one thread is used.
                    pass
                
            flat_trace = sampler.chain[:, burn::thin, :]
            flat_trace = flat_trace.reshape((-1, flat_trace.shape[2]))
        else:
            flat_trace = flat_trace[burn::thin, :]
        
        num_proc = kwargs.get('num_proc', multiprocessing.cpu_count())
        
        if num_proc > 1:
            pool = InterruptiblePool(processes=num_proc)
            map_fun = pool.map
        else:
            map_fun = map
        try:
            res = map_fun(
                _ComputeGPWrapper(
                    self,
                    X,
                    n,
                    return_mean,
                    return_std,
                    return_cov,
                    return_samples,
                    return_mean_func,
                    num_samples,
                    noise,
                    samp_kwargs,
                    output_transform
                ),
                flat_trace
            )
        finally:
            if num_proc > 1:
                pool.close()
        out = dict()
        if return_mean:
            out['mean'] = [r['mean'] for r in res if r is not None]
        if return_std:
            out['std'] = [r['std'] for r in res if r is not None]
        if return_cov:
            out['cov'] = [r['cov'] for r in res if r is not None]
        if return_samples:
            out['samp'] = [r['samp'] for r in res if r is not None]
        if return_mean_func and self.mu is not None:
            out['mean_func'] = [r['mean_func'] for r in res if r is not None]
            out['cov_func'] = [r['cov_func'] for r in res if r is not None]
            out['std_func'] = [r['std_func'] for r in res if r is not None]
            
            out['mean_without_func'] = [r['mean_without_func'] for r in res if r is not None]
            out['cov_without_func'] = [r['cov_without_func'] for r in res if r is not None]
            out['std_without_func'] = [r['std_without_func'] for r in res if r is not None]
        return out