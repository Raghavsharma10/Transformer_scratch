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
        if len(p) != len(self.bounds):
            raise ValueError("length of p must equal the number of parameters!")
        if p.ndim != 1:
            raise ValueError("p must be one-dimensional!")
        c = scipy.zeros(len(self.bounds))
        
        # Old way, based on sorted uniform variables:
        # for k in xrange(0, len(self.bounds)):
        #     if p[k] <= self.bounds[k][0]:
        #         c[k] = 0.0
        #     elif p[k] >= self.bounds[k][1]:
        #         c[k] = 1.0
        #     else:
        #         c[k] = (p[k] - self.bounds[k][0]) / (self.bounds[k][1] - self.bounds[k][0])
        
        # New way, based on conditional marginals:
        for d in xrange(0, len(c)):
            pdm1 = p[d - 1] if d > 0 else self.lb
            if p[d] <= pdm1:
                c[d] = 0.0
            elif p[d] >= self.ub:
                c[d] = 1.0
            else:
                c[d] = 1.0 - (1.0 - (p[d] - pdm1) / (self.ub - pdm1))**(self.num_var - d)
        
        return c