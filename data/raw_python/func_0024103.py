def elementwise_cdf(self, p):
        r"""Convert a sample to random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the CDF. To
        facilitate efficient sampling, this function returns a *vector* of CDF
        values, one value for each variable. Basically, the idea is that, given
        a vector :math:`q` of `num_params` values each of which is distributed
        according to the prior, this function will return variables uniform on
        :math:`[0, 1]` corresponding to each variable. This is the inverse
        operation to :py:meth:`sample_u`.
        
        Parameters
        ----------
        p : array-like, (`num_params`,)
            Values to evaluate CDF at.
        """
        p = scipy.atleast_1d(p)
        if len(p) != len(self.a):
            raise ValueError("length of p must equal the number of parameters!")
        if p.ndim != 1:
            raise ValueError("p must be one-dimensional!")
        return scipy.asarray([scipy.stats.gamma.cdf(v, a, loc=0, scale=1.0 / b) for v, a, b in zip(p, self.a, self.b)])