def plot(self,minval=None,maxval=None,fig=None,log=False,
             npts=500,**kwargs):
        """
        Plots distribution.

        Parameters
        ----------
        minval : float,optional
            minimum value to plot.  Required if minval of Distribution is 
            `-np.inf`.

        maxval : float, optional
            maximum value to plot.  Required if maxval of Distribution is 
            `np.inf`.

        fig : None or int, optional
            Parameter to pass to `setfig`.  If `None`, then a new figure is 
            created; if a non-zero integer, the plot will go to that figure 
            (clearing everything first), if zero, then will overplot on 
            current axes.

        log : bool, optional
            If `True`, the x-spacing of the points to plot will be logarithmic.

        npoints : int, optional
            Number of points to plot.

        kwargs
            Keyword arguments are passed to plt.plot

        Raises
        ------
        ValueError
            If finite lower and upper bounds are not provided.
        """
        if minval is None:
            minval = self.minval
        if maxval is None:
            maxval = self.maxval
        if maxval==np.inf or minval==-np.inf:
            raise ValueError('must have finite upper and lower bounds to plot. (use minval, maxval kws)')

        if log:
            xs = np.logspace(np.log10(minval),np.log10(maxval),npts)
        else:
            xs = np.linspace(minval,maxval,npts)

        setfig(fig)
        plt.plot(xs,self(xs),**kwargs)
        plt.xlabel(self.name)
        plt.ylim(ymin=0,ymax=self(xs).max()*1.2)