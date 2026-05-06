def from_optimize_result(cls, result, n, m, index=None):
        """Construct a Projection from the output of an optimization.

        Args:
            result (:py:class:`scipy.optimize.OptimizeResult`): Object 
                returned by :py:func:`scipy.optimize.minimize`.
            n (`int`): Number of dimensions.
            m (`int`): Number of samples.
            index (`list-like`): Names of samples. (Optional).

        Returns:
            :py:class:`pymds.Projection`
        """
        coords = pd.DataFrame(result.x.reshape((m, n)), index=index)
        projection = cls(coords)
        projection.stress = result.fun
        return projection