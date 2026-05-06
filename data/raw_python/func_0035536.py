def d_distributionparams(self):
        """See docs for `DistributionModel` abstract base class."""
        if not self._d_distributionparams:
            dx = 1.0e-3
            def f_alpha(alpha):
                return DiscreteGamma(alpha, self.beta_lambda, self.ncats)
            def f_beta(beta):
                return DiscreteGamma(self.alpha_lambda, beta, self.ncats)
            assert set(self.distributionparams) == {'alpha_lambda', 'beta_lambda'}
            for (param, f) in [('alpha_lambda', f_alpha), ('beta_lambda', f_beta)]:
                pvalue = getattr(self, param)
                dparam = scipy.misc.derivative(f, pvalue, dx, n=1, order=5)
                assert dparam.shape == (self.ncats,)
                for stepchange in [0.5, 2]: # make sure robust to step size
                    dparam2 = scipy.misc.derivative(f, pvalue, stepchange * dx,
                            n=1, order=5)
                    assert scipy.allclose(dparam, dparam2, atol=1e-5, rtol=1e-4), (
                            "Numerical derivative of {0} at {1} "
                            "differs for step {2} and {3}: {4} and {5}"
                            ", respectively.").format(param, pvalue,
                            dx, dx * stepchange, dparam, dparam2)
                self._d_distributionparams[param] = dparam
        return self._d_distributionparams