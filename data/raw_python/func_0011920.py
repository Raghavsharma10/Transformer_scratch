def _minimize_scalar(
        self, desc="Progress", rtol=1.4902e-08, atol=1.4902e-08, verbose=True
    ):
        """
        Minimize a scalar function using Brent's method.

        Parameters
        ----------
        verbose : bool
            ``True`` for verbose output; ``False`` otherwise.
        """
        from tqdm import tqdm
        from numpy import asarray
        from brent_search import minimize as brent_minimize

        variables = self._variables.select(fixed=False)
        if len(variables) != 1:
            raise ValueError("The number of variables must be equal to one.")

        var = variables[variables.names()[0]]
        progress = tqdm(desc=desc, disable=not verbose)

        def func(x):
            progress.update(1)
            var.value = x
            return self.__sign * self.value()

        r = asarray(
            brent_minimize(func, a=var.bounds[0], b=var.bounds[1], rtol=rtol, atol=atol)
        )
        var.value = r[0]
        progress.close()