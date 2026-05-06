def resample(self,N,minval=None,maxval=None,log=False,res=1e4):
        """Returns random samples generated according to the distribution

        Mirrors basic functionality of `rvs` method for `scipy.stats`
        random variates.  Implemented by mapping uniform numbers onto the
        inverse CDF using a closest-matching grid approach.

        Parameters
        ----------
        N : int
            Number of samples to return

        minval,maxval : float, optional
            Minimum/maximum values to resample.  Should both usually just be 
            `None`, which will default to `self.minval`/`self.maxval`.

        log : bool, optional
            Whether grid should be log- or linear-spaced.

        res : int, optional
            Resolution of CDF grid used.

        Returns
        -------
        values : ndarray
            N samples.

        Raises
        ------
        ValueError
            If maxval/minval are +/- infinity, this doesn't work because of
            the grid-based approach.

        """
        N = int(N)
        if minval is None:
            if hasattr(self,'minval_cdf'):
                minval = self.minval_cdf
            else:
                minval = self.minval
        if maxval is None:
            if hasattr(self,'maxval_cdf'):
                maxval = self.maxval_cdf
            else:
                maxval = self.maxval

        if maxval==np.inf or minval==-np.inf:
            raise ValueError('must have finite upper and lower bounds to resample. (set minval, maxval kws)')

        u = rand.random(size=N)
        if log:
            vals = np.logspace(log10(minval),log10(maxval),res)
        else:
            vals = np.linspace(minval,maxval,res)
            
        #sometimes cdf is flat.  so ys will need to be uniqued
        ys,yinds = np.unique(self.cdf(vals), return_index=True)
        vals = vals[yinds]
        

        inds = np.digitize(u,ys)
        return vals[inds]