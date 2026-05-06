def sample_hyperparameter_posterior(self, nwalkers=200, nsamp=500, burn=0,
                                        thin=1, num_proc=None, sampler=None,
                                        plot_posterior=False,
                                        plot_chains=False, sampler_type='ensemble',
                                        ntemps=20, sampler_a=2.0, **plot_kwargs):
        """Produce samples from the posterior for the hyperparameters using MCMC.
        
        Returns the sampler created, because storing it stops the GP from being
        pickleable. To add more samples to a previous sampler, pass the sampler
        instance in the `sampler` keyword.
        
        Parameters
        ----------
        nwalkers : int, optional
            The number of walkers to use in the sampler. Should be on the order
            of several hundred. Default is 200.
        nsamp : int, optional
            Number of samples (per walker) to take. Default is 500.
        burn : int, optional
            This keyword only has an effect on the corner plot produced when
            `plot_posterior` is True and the flattened chain plot produced
            when `plot_chains` is True. To perform computations with burn-in,
            see :py:meth:`compute_from_MCMC`. The number of samples to discard
            at the beginning of the chain. Default is 0.
        thin : int, optional
            This keyword only has an effect on the corner plot produced when
            `plot_posterior` is True and the flattened chain plot produced
            when `plot_chains` is True. To perform computations with thinning,
            see :py:meth:`compute_from_MCMC`. Every `thin`-th sample is kept.
            Default is 1.
        num_proc : int or None, optional
            Number of processors to use. If None, all available processors are
            used. Default is None (use all available processors).
        sampler : :py:class:`Sampler` instance
            The sampler to use. If the sampler already has samples, the most
            recent sample will be used as the starting point. Otherwise a
            random sample from the hyperprior will be used.
        plot_posterior : bool, optional
            If True, a corner plot of the posterior for the hyperparameters
            will be generated. Default is False.
        plot_chains : bool, optional
            If True, a plot showing the history and autocorrelation of the
            chains will be produced.
        sampler_type : str, optional
            The type of sampler to use. Valid options are "ensemble" (affine-
            invariant ensemble sampler) and "pt" (parallel-tempered ensemble
            sampler).
        ntemps : int, optional
            Number of temperatures to use with the parallel-tempered ensemble
            sampler.
        sampler_a : float, optional
            Scale of the proposal distribution.
        plot_kwargs : additional keywords, optional
            Extra arguments to pass to :py:func:`~gptools.utils.plot_sampler`.
        """
        if num_proc is None:
            num_proc = multiprocessing.cpu_count()
        # Needed for emcee to do it right:
        if num_proc == 0:
            num_proc = 1
        ndim = len(self.free_params)
        if sampler is None:
            if sampler_type == 'ensemble':
                sampler = emcee.EnsembleSampler(
                    nwalkers,
                    ndim,
                    _ComputeLnProbEval(self),
                    threads=num_proc,
                    a=sampler_a
                )
            elif sampler_type == 'pt':
                # TODO: Finish this!
                raise NotImplementedError("PTSampler not done yet!")
                sampler = emcee.PTSampler(
                    ntemps,
                    nwalkers,
                    ndim,
                    logl,
                    logp
                )
            else:
                raise NotImplementedError(
                    "Sampler type %s not supported!" % (sampler_type,)
                )
        else:
            sampler.a = sampler_a
        if sampler.chain.size == 0:
            theta0 = self.hyperprior.random_draw(size=nwalkers).T
            theta0 = theta0[:, ~self.fixed_params]
        else:
            # Start from the stopping point of the previous chain:
            theta0 = sampler.chain[:, -1, :]
        
        sampler.run_mcmc(theta0, nsamp)
        if plot_posterior or plot_chains:
            flat_trace = sampler.chain[:, burn::thin, :]
            flat_trace = flat_trace.reshape((-1, flat_trace.shape[2]))
        
        if plot_posterior and plot_chains:
            plot_sampler(
                sampler,
                labels=['$%s$' % (l,) for l in self.free_param_names],
                burn=burn,
                **plot_kwargs
            )
        else:
            if plot_posterior:
                triangle.corner(
                    flat_trace,
                    plot_datapoints=False,
                    labels=['$%s$' % (l,) for l in self.free_param_names]
                )
            if plot_chains:
                f = plt.figure()
                for k in xrange(0, ndim):
                    # a = f.add_subplot(3, ndim, k + 1)
                    # a.acorr(
                    #     sampler.flatchain[:, k],
                    #     maxlags=100,
                    #     detrend=plt.mlab.detrend_mean
                    # )
                    # a.set_xlabel('lag')
                    # a.set_title('$%s$ autocorrelation' % (self.free_param_names[k],))
                    a = f.add_subplot(ndim, 1, 0 * ndim + k + 1)
                    for chain in sampler.chain[:, :, k]:
                        a.plot(chain)
                    a.set_xlabel('sample')
                    a.set_ylabel('$%s$' % (self.free_param_names[k],))
                    a.set_title('$%s$ all chains' % (self.free_param_names[k],))
                    a.axvline(burn, color='r', linewidth=3, ls='--')
                    # a = f.add_subplot(2, ndim, 1 * ndim + k + 1)
                    # a.plot(flat_trace[:, k])
                    # a.set_xlabel('sample')
                    # a.set_ylabel('$%s$' % (self.free_param_names[k],))
                    # a.set_title('$%s$ flattened, burned and thinned chain' % (self.free_param_names[k],))
        
        # Print a summary of the sampler:
        print("MCMC parameter summary:")
        print("param\tmean\t95% posterior interval")
        mean, ci_l, ci_u = summarize_sampler(sampler, burn=burn)
        names = self.free_param_names[:]
        for n, m, l, u in zip(names, mean, ci_l, ci_u):
            print("%s\t%4.4g\t[%4.4g, %4.4g]" % (n, m, l, u))
        
        return sampler