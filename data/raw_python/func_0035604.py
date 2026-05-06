def setCovariance(self, cov):
        """ makes lowrank approximation of cov """
        assert cov.shape[0]==self.dim, 'Dimension mismatch.'
        S, U = la.eigh(cov)
        U = U[:,::-1]
        S = S[::-1]
        _X = U[:, :self.rank] * sp.sqrt(S[:self.rank])
        self.X = _X