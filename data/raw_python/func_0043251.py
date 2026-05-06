def optimize(self, start=None, n=2):
        """Run multidimensional scaling on this distance matrix.

        Args:
            start (`None` or `array-like`): Starting coordinates. If
                `start=None`, random starting coordinates are used. If
                `array-like` must have shape [`m` * `n`, ].
            n (`int`): Number of dimensions to embed samples in.

        Examples:

            .. doctest::

               >>> import pandas as pd
               >>> from pymds import DistanceMatrix
               >>> dist = pd.DataFrame({
               ...    'a': [0.0, 1.0, 2.0],
               ...    'b': [1.0, 0.0, 3 ** 0.5],
               ...    'c': [2.0, 3 ** 0.5, 0.0]} , index=['a', 'b', 'c'])
               >>> dm = DistanceMatrix(dist)
               >>> pro = dm.optimize(n=2)
               >>> pro.coords.shape
               (3, 2)
               >>> type(pro)
               <class 'pymds.mds.Projection'>


        Returns: :py:class:`pymds.Projection`
        """
        self.n = n

        if start is None:
            start = np.random.rand(self.m * self.n) * 10

        optim = minimize(
            fun=self._error_and_gradient,
            x0=start,
            jac=True,
            method='L-BFGS-B')

        index = self.index if hasattr(self, "index") else None

        return Projection.from_optimize_result(
            result=optim, n=self.n, m=self.m, index=index)