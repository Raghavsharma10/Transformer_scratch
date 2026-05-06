def fit(self, fitfunction, parinit, unfittableparameters=(), *args, **kwargs):
        """Perform a nonlinear least-squares fit, using sastool.misc.fitter.Fitter()

        Other arguments and keyword arguments will be passed through to the
        __init__ method of Fitter. For example, these are:
        - lbounds
        - ubounds
        - ytransform
        - loss
        - method

        Returns: the final parameters as ErrorValue instances, the stats
            dictionary and the fitted curve instance of the same type as
            this)
        """
        kwargs['otherparameters'] = unfittableparameters
        fitter = Fitter(fitfunction, parinit, self.q, self.Intensity, self.qError, self.Error, *args, **kwargs)
        fixedvalues = [[None, p][isinstance(p, FixedParameter)] for p in parinit]
        fitter.fixparameters(fixedvalues)
        fitter.fit()
        pars = fitter.parameters()
        uncs = fitter.uncertainties()
        stats = fitter.stats()
        results = [ErrorValue(p, u) for p, u in zip(pars, uncs)] + [stats, type(self)(self.q, stats['func_value'])]
        return results