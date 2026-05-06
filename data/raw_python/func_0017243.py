def cov(self, ddof=None, bias=0):
        '''The covariance matrix from the aggregate sample. It accepts an
        optional parameter for the degree of freedoms.

        :parameter ddof: If not ``None`` normalization is by (N - ddof), where N is
            the number of observations; this overrides the value implied by bias.
            The default value is None.
        '''
        N = self.n
        M = N if bias else N-1
        M = M if ddof is None else N-ddof
        return (self.sxx - outer(self.sx,self.sx)/N)/M