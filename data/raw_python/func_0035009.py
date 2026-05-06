def maximizeLikelihood(self, optimize_brlen=False,
            approx_grad=False, logliktol=1.0e-2, nparamsretry=1,
            printfunc=None):
        """Maximize the log likelihood.

        Maximizes log likelihood with respect to model parameters
        and potentially branch lengths depending on `optimize_brlen`.
        If optimizing the branch lengths, iterates between optimizing
        the model parameters and branch lengths.

        Uses the L-BFGS-B method implemented in `scipy.optimize`.

        There is no return variable, but after call object attributes
        will correspond to maximimum likelihood values.

        Args:
            `optimize_brlen` (bool)
                Do we optimize branch lengths?
            `approx_grad` (bool)
                If `True`, then we numerically approximate the gradient
                rather than using the analytical values.
            `logliktol` (float)
                When using `optimize_brlen`, keep iterating between
                optimization of parameters and branch lengths until
                change in log likelihood is less than `logliktol`.
            `nparamsretry` (int >= 0)
                Number of times to retry parameter optimization from
                different initial values if it fails the first time.
            `printfunc` (`None` or a function)
                If not `None`, then we print using `printfunc` the
                detailed results of the optimization at each step.
                For instance, `printfunc` might be `sys.stderr.write`
                or `logger.info`.

        Returns:
            A string giving a summary of the maximization.
        """
        # Some useful notes on optimization:
        # http://www.scipy-lectures.org/advanced/mathematical_optimization/

        assert len(self.paramsarray) > 0, "No parameters to optimize"
        assert nparamsretry >= 0
        assert logliktol > 0

        def paramsfunc(x):
            """Negative log likelihood when `x` is params."""
            self.paramsarray = x
            return -self.loglik

        def paramsdfunc(x):
            """Negative gradient log likelihood with respect to params."""
            self.paramsarray = x
            return -self.dloglikarray

        def tfunc(x):
            """Negative log likelihood when `x` is branch lengths."""
            self.t = x
            return -self.loglik

        def tdfunc(x):
            """Negative gradient loglik with respect to branch lengths."""
            self.t = x
            return -self.dloglik_dt

        if approx_grad:
            paramsdfunc = False
            tdfunc = False
            self.dtcurrent = False
            self.dparamscurrent = False

        def _printResult(opttype, result, i, old, new):
            """Print summary of optimization result."""
            if printfunc is not None:
                printfunc('Step {0}, optimized {1}.\n'
                          'Likelihood went from {2} to {3}.\n'
                          'Max magnitude in Jacobian is {4}.\n'
                          'Full optimization result:\n{5}\n'.format(
                          i, opttype, old, new,
                          scipy.absolute(result.jac).max(), result))

        oldloglik = self.loglik
        converged = False
        firstbrlenpass = True
        options = {'ftol':1.0e-7} # optimization options
        summary = []
        i = 1
        while not converged:
            if (not self.dparamscurrent) and (not approx_grad):
                self.dtcurrent = False
                self.dparamscurrent = True
            nparamstry = 0
            origparamsarray = self.paramsarray.copy()
            paramsconverged = False
            while not paramsconverged:
                result = scipy.optimize.minimize(paramsfunc, self.paramsarray,
                        method='L-BFGS-B', jac=paramsdfunc,
                        bounds=self.paramsarraybounds, options=options)
                _printResult('params', result, i, oldloglik, self.loglik)
                msg = ('Step {0}: optimized parameters, loglik went from '
                        '{1:.2f} to {2:.2f} ({3} iterations, {4} function '
                        'evals)'.format(i, oldloglik, self.loglik, result.nit,
                        result.nfev))
                summary.append(msg)
                if result.success and (not (oldloglik - self.loglik > logliktol)):
                    paramsconverged = True
                    jacmax = scipy.absolute(result.jac).max()
                    if (jacmax > 1000) and not (firstbrlenpass and optimize_brlen):
                        warnings.warn("Optimizer reports convergence, "
                                "but max element in Jacobian is {0}\n"
                                "Summary of optimization:\n{1}".format(
                                jacmax, summary))
                else:
                    if not result.success:
                        resultmessage = result.message
                    else:
                        resultmessage = ('loglik increased in param optimization '
                                'from {0} to {1}'.format(oldloglik, self.loglik))
                    nparamstry += 1
                    failmsg = ("Optimization failure {0}\n{1}\n{2}".format(
                            nparamstry, resultmessage, '\n'.join(summary)))
                    if nparamstry > nparamsretry:
                        raise RuntimeError(failmsg)
                    else:
                        warnings.warn(failmsg + '\n\n' +
                                "Re-trying with different initial params.")
                        scipy.random.seed(nparamstry)
                        # seed at geometric mean of original value, max
                        # bound, min bound, and random number between max and min
                        minarray = scipy.array([self.paramsarraybounds[j][0] for
                                j in range(len(self.paramsarray))])
                        maxarray = scipy.array([self.paramsarraybounds[j][1] for
                                j in range(len(self.paramsarray))])
                        randarray = scipy.random.uniform(minarray, maxarray)
                        newarray = (minarray * maxarray * randarray *
                                origparamsarray)**(1 / 4.) # geometric mean
                        assert newarray.shape == self.paramsarray.shape
                        assert (newarray > minarray).all()
                        assert (newarray < maxarray).all()
                        self.paramsarray = newarray
            i += 1
            assert oldloglik - self.loglik <= logliktol
            if (self.loglik - oldloglik >= logliktol) or firstbrlenpass:
                firstbrlenpass = False
                oldloglik = self.loglik
                if optimize_brlen:
                    if not approx_grad:
                        self.dparamscurrent = False
                        self.dtcurrent = True
                    result = scipy.optimize.minimize(tfunc, self.t,
                            method='L-BFGS-B', jac=tdfunc, options=options,
                            bounds=[(ALMOST_ZERO, None)] * len(self.t))
                    _printResult('branches', result, i, oldloglik, self.loglik)
                    summary.append('Step {0}: optimized branches, loglik '
                            'went from {1:.2f} to {2:.2f} ({3} iterations, '
                            '{4} function evals)'.format(i, oldloglik,
                            self.loglik, result.nit, result.nfev))
                    i += 1
                    assert result.success, ("Optimization failed\n{0}"
                            "\n{1}\n{2}".format(result.message, self.t,
                            '\n'.join(summary)))
                    if oldloglik - self.loglik > logliktol:
                        raise RuntimeError("loglik increased during t "
                                "optimization: {0} to {1}".format(
                                oldloglik, self.loglik))
                    elif self.loglik - oldloglik >= logliktol:
                        oldloglik = self.loglik
                    else:
                        converged = True
                else:
                    converged = True
            else:
                converged = True

        return '\n'.join(summary)