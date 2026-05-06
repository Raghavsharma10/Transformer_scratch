def compute_l_from_MCMC(self, X, n=0, sampler=None, flat_trace=None, burn=0, thin=1, **kwargs):
        """Compute desired quantities from MCMC samples of the hyperparameter posterior.
        
        The return will be a list with a number of rows equal to the number of
        hyperparameter samples. The columns will contain the covariance length
        scale function.
        
        Parameters
        ----------
        X : array-like (`M`,) or (`M`, `num_dim`)
            The values to evaluate the Gaussian process at.
        n : non-negative int or list, optional
            The order of derivative to compute. For num_dim=1, this must be an
            int. For num_dim=2, this must be a list of ints of length 2.
            Default is 0 (don't take derivative).
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
        out : array of float
            Length scale function at the indicated points.
        """
        if flat_trace is None:
            if sampler is None:
                sampler = self.sample_hyperparameter_posterior(burn=burn, **kwargs)
                # If we create the sampler, we need to make sure we clean up
                # its pool:
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
            try:
                res = pool.map(_ComputeLWrapper(self, X, n), flat_trace)
            finally:
                pool.close()
        else:
            res = map(_ComputeLWrapper(self, X, n), flat_trace)
        
        return res