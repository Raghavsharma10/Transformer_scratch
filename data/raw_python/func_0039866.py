def pctile(self,pct,res=1000):
        """Returns the desired percentile of the distribution.

        Will only work if properly normalized.  Designed to mimic
        the `ppf` method of the `scipy.stats` random variate objects.
        Works by gridding the CDF at a given resolution and matching the nearest
        point.  NB, this is of course not as precise as an analytic ppf.

        Parameters
        ----------

        pct : float
            Percentile between 0 and 1.

        res : int, optional
            The resolution at which to grid the CDF to find the percentile.

        Returns
        -------
        percentile : float
        """
        grid = np.linspace(self.minval,self.maxval,res)
        return grid[np.argmin(np.absolute(pct-self.cdf(grid)))]