def corr(self):
        '''The correlation matrix'''
        cov = self.cov()
        N = cov.shape[0]
        corr = ndarray((N,N))
        for r in range(N):
            for c in range(r):
                corr[r,c] = corr[c,r] = cov[r,c]/sqrt(cov[r,r]*cov[c,c])
            corr[r,r] = 1.
        return corr