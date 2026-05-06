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
        p1_num_params = len(self.p1.bounds)
        return scipy.concatenate(
            (
                self.p1.elementwise_cdf(p[:p1_num_params]),
                self.p2.elementwise_cdf(p[p1_num_params:])
            )
        )