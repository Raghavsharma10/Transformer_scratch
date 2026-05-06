def sample_u(self, q):
        r"""Extract a sample from random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the inverse
        CDF. To facilitate efficient sampling, this function returns a *vector*
        of PPF values, one value for each variable. Basically, the idea is that,
        given a vector :math:`q` of `num_params` values each of which is
        distributed uniformly on :math:`[0, 1]`, this function will return
        corresponding samples for each variable.
        
        Parameters
        ----------
        q : array of float
            Values between 0 and 1 to evaluate inverse CDF at.
        """
        q = scipy.atleast_1d(q)
        if len(q) != self.num_var:
            raise ValueError("length of q must equal the number of parameters!")
        if q.ndim != 1:
            raise ValueError("q must be one-dimensional!")
        if (q < 0).any() or (q > 1).any():
            raise ValueError("q must be within [0, 1]!")
        
        # Old way, not quite correct:
        # q = scipy.sort(q)
        # return scipy.asarray([(self.ub - self.lb) * v + self.lb for v in q])
        
        # New way, based on conditional marginals:
        out = scipy.zeros_like(q, dtype=float)
        out[0] = self.lb
        for d in xrange(0, len(out)):
            out[d] = (
                (1.0 - (1.0 - q[d])**(1.0 / (self.num_var - d))) *
                (self.ub - out[max(d - 1, 0)]) + out[max(d - 1, 0)]
            )
        return out