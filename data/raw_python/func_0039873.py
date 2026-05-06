def resample(self,N,**kwargs):
        """Random resampling of the doublegauss distribution
        """
        lovals = self.mu - np.absolute(rand.normal(size=N)*self.siglo)
        hivals = self.mu + np.absolute(rand.normal(size=N)*self.sighi)

        u = rand.random(size=N)
        hi = (u < float(self.sighi)/(self.sighi + self.siglo))
        lo = (u >= float(self.sighi)/(self.sighi + self.siglo))

        vals = np.zeros(N)
        vals[hi] = hivals[hi]
        vals[lo] = lovals[lo]
        return vals