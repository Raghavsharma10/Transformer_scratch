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
        q : array-like, (`num_params`,)
            Values between 0 and 1 to evaluate inverse CDF at.
        """
        p1_num_params = len(self.p1.bounds)
        return scipy.concatenate(
            (
                self.p1.sample_u(q[:p1_num_params]),
                self.p2.sample_u(q[p1_num_params:])
            )
        )