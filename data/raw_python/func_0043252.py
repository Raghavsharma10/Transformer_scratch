def optimize_batch(self, batchsize=10, returns='best', paralell=True):
        """
        Run multiple optimizations using different starting coordinates.

        Args:
            batchsize (`int`): Number of optimizations to run.
            returns (`str`): If ``'all'``, return results of all optimizations,
                ordered by stress, ascending. If ``'best'`` return the
                projection with the lowest stress.
            parallel (`bool`): If ``True``, run optimizations in parallel.

        Examples:

            .. doctest::

               >>> import pandas as pd
               >>> from pymds import DistanceMatrix
               >>> dist = pd.DataFrame({
               ...    'a': [0.0, 1.0, 2.0],
               ...    'b': [1.0, 0.0, 3 ** 0.5],
               ...    'c': [2.0, 3 ** 0.5, 0.0]} , index=['a', 'b', 'c'])
               >>> dm = DistanceMatrix(dist)
               >>> batch = dm.optimize_batch(batchsize=3, returns='all')
               >>> len(batch)
               3
               >>> type(batch[0])
               <class 'pymds.mds.Projection'>

        Returns:
            `list` or :py:class:`pymds.Projection`:

                `list`: Length batchsize, containing instances of
                :py:class:`pymds.Projection`. Sorted by stress, ascending.

                or

                :py:class:`pymds.Projection`: Projection with the lowest
                stress.
        """
        if returns not in ('best', 'all'):
            raise ValueError('returns must be either "best" or "all"')

        starts = [np.random.rand(self.m * 2) * 10 for i in range(batchsize)]

        if paralell:
            with Pool() as p:
                results = p.map(self.optimize, starts)
        else:
            results = map(self.optimize, starts)

        results = sorted(results, key=lambda x: x.stress)

        return results if returns == 'all' else results[0]