def plothist(self,fig=None,**kwargs):
        """Plots a histogram of samples using provided bins.
        
        Parameters
        ----------
        fig : None or int
            Parameter passed to `setfig`.

        kwargs
            Keyword arguments passed to `plt.hist`.
        """
        setfig(fig)
        plt.hist(self.samples,bins=self.bins,**kwargs)