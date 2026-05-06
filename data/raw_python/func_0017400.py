def var(self, ddof=0):
        '''Calculate variance of timeseries. Return a vector containing
the variances of each series in the timeseries.

:parameter ddof: delta degree of freedom, the divisor used in the calculation
                 is given by ``N - ddof`` where ``N`` represents the length
                 of timeseries. Default ``0``.

.. math::

    var = \\frac{\\sum_i^N (x - \\mu)^2}{N-ddof}
    '''
        N = len(self)
        if N:
            v = self.values()
            mu = sum(v)
            return (sum(v*v) - mu*mu/N)/(N-ddof)
        else:
            return None