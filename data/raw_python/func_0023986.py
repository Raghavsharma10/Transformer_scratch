def optimize_hyperparameters(self, method='SLSQP', opt_kwargs={},
                                 verbose=False, random_starts=None,
                                 num_proc=None, max_tries=1):
        r"""Optimize the hyperparameters by maximizing the log-posterior.
        
        Leaves the :py:class:`GaussianProcess` instance in the optimized state.
        
        If :py:func:`scipy.optimize.minimize` is not available (i.e., if your
        :py:mod:`scipy` version is older than 0.11.0) then :py:func:`fmin_slsqp`
        is used independent of what you set for the `method` keyword.
        
        If :py:attr:`use_hyper_deriv` is True the optimizer will attempt to use
        the derivatives of the log-posterior with respect to the hyperparameters
        to speed up the optimization. Note that only the squared exponential
        covariance kernel supports hyperparameter derivatives at present.
        
        Parameters
        ----------
        method : str, optional
            The method to pass to :py:func:`scipy.optimize.minimize`.
            Refer to that function's docstring for valid options. Default
            is 'SLSQP'. See note above about behavior with older versions of
            :py:mod:`scipy`.
        opt_kwargs : dict, optional
            Dictionary of extra keywords to pass to
            :py:func:`scipy.optimize.minimize`. Refer to that function's
            docstring for valid options. Default is: {}.
        verbose : bool, optional
            Whether or not the output should be verbose. If True, the entire
            :py:class:`Result` object from :py:func:`scipy.optimize.minimize` is
            printed. If False, status information is only printed if the
            `success` flag from :py:func:`minimize` is False. Default is False.
        random_starts : non-negative int, optional
            Number of times to randomly perturb the starting guesses
            (distributed according to the hyperprior) in order to seek the
            global minimum. If None, then `num_proc` random starts will be
            performed. Default is None (do number of random starts equal to the
            number of processors allocated). Note that for `random_starts` != 0,
            the initial state of the hyperparameters is not actually used.
        num_proc : non-negative int or None, optional
            Number of processors to use with random starts. If 0, processing is
            not done in parallel. If None, all available processors are used.
            Default is None (use all available processors).
        max_tries : int, optional
            Number of times to run through the random start procedure if a
            solution is not found. Default is to only go through the procedure
            once.
        """
        if opt_kwargs is None:
            opt_kwargs = {}
        else:
            opt_kwargs = dict(opt_kwargs)
        if 'method' in opt_kwargs:
            method = opt_kwargs['method']
            if self.verbose:
                warnings.warn(
                    "Key 'method' is present in opt_kwargs, will override option "
                    "specified with method kwarg.",
                    RuntimeWarning
                )
        else:
            opt_kwargs['method'] = method
        
        if num_proc is None:
            num_proc = multiprocessing.cpu_count()
        
        param_ranges = scipy.asarray(self.free_param_bounds, dtype=float)
        # Replace unbounded variables with something big:
        param_ranges[scipy.where(scipy.isnan(param_ranges[:, 0])), 0] = -1e16
        param_ranges[scipy.where(scipy.isnan(param_ranges[:, 1])), 1] = 1e16
        param_ranges[scipy.where(scipy.isinf(param_ranges[:, 0])), 0] = -1e16
        param_ranges[scipy.where(scipy.isinf(param_ranges[:, 1])), 1] = 1e16
        if random_starts == 0:
            num_proc = 0
            param_samples = [self.free_params[:]]
        else:
            if random_starts is None:
                random_starts = max(num_proc, 1)
            # Distribute random guesses according to the hyperprior:
            param_samples = self.hyperprior.random_draw(size=random_starts).T
            param_samples = param_samples[:, ~self.fixed_params]
        if 'bounds' not in opt_kwargs:
            opt_kwargs['bounds'] = param_ranges
        if self.use_hyper_deriv:
            opt_kwargs['jac'] = True
        trial = 0
        res_min = None
        while trial < max_tries and res_min is None:
            if trial >= 1:
                if self.verbose:
                    warnings.warn(
                        "No solutions found on trial %d, retrying random starts." % (trial - 1,),
                        RuntimeWarning
                    )
                # Produce a new initial guess:
                if random_starts != 0:
                    param_samples = self.hyperprior.random_draw(size=random_starts).T
                    param_samples = param_samples[:, ~self.fixed_params]
            trial += 1
            if num_proc > 1:
                pool = InterruptiblePool(processes=num_proc)
                map_fun = pool.map
            else:
                map_fun = map
            try:
                res = map_fun(
                    _OptimizeHyperparametersEval(self, opt_kwargs),
                    param_samples
                )
            finally:
                if num_proc > 1:
                    pool.close()
            # Filter out the failed convergences:
            res = [r for r in res if r is not None]
            
            try:
                res_min = min(res, key=lambda r: r.fun)
                if scipy.isnan(res_min.fun) or scipy.isinf(res_min.fun):
                    res_min = None
            except ValueError:
                res_min = None
        
        if res_min is None:
            raise ValueError(
                "Optimizer failed to find a valid solution. Try changing the "
                "parameter bounds, picking a new initial guess or increasing the "
                "number of random starts."
            )
        
        self.update_hyperparameters(res_min.x)
        if verbose:
            print("Got %d completed starts, optimal result is:" % (len(res),))
            print(res_min)
            print("\nLL\t%.3g" % (-1 * res_min.fun))
            for v, l in zip(res_min.x, self.free_param_names):
                print("%s\t%.3g" % (l.translate(None, '\\'), v))
        if not res_min.success:
            warnings.warn(
                "Optimizer %s reports failure, selected hyperparameters are "
                "likely NOT optimal. Status: %d, Message: '%s'. Try adjusting "
                "bounds, initial guesses or the number of random starts used."
                % (
                    method,
                    res_min.status,
                    res_min.message
                ),
                RuntimeWarning
            )
        bounds = scipy.asarray(self.free_param_bounds)
        # Augment the bounds a little bit to catch things that are one step away:
        if ((res_min.x <= 1.001 * bounds[:, 0]).any() or
            (res_min.x >= 0.999 * bounds[:, 1]).any()):
            warnings.warn(
                "Optimizer appears to have hit/exceeded the bounds. Bounds are:\n"
                "%s\n, solution is:\n%s. Try adjusting bounds, initial guesses "
                "or the number of random starts used."
                % (str(bounds), str(res_min.x),)
            )
        return (res_min, len(res))