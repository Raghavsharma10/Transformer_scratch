def plot(self, hist=False, show=False, **kwargs):
        """
        Plot the distribution of the UncertainVariable. Continuous 
        distributions are plotted with a line plot and discrete distributions
        are plotted with discrete circles.
        
        Optional
        --------
        hist : bool
            If true, a histogram is displayed
        show : bool
            If ``True``, the figure will be displayed after plotting the 
            distribution. If ``False``, an explicit call to ``plt.show()`` is
            required to display the figure.
        kwargs : any valid matplotlib.pyplot.plot kwarg
        
        """
        import matplotlib.pyplot as plt

        if hist:
            vals = self._mcpts
            low = vals.min()
            high = vals.max()
            h = plt.hist(
                vals,
                bins=int(np.sqrt(len(vals)) + 0.5),
                histtype="stepfilled",
                normed=True,
                **kwargs
            )
            plt.ylim(0, 1.1 * h[0].max())
        else:
            bound = 0.0001
            low = self.rv.ppf(bound)
            high = self.rv.ppf(1 - bound)
            if hasattr(self.rv.dist, "pmf"):
                low = int(low)
                high = int(high)
                vals = list(range(low, high + 1))
                plt.plot(vals, self.rv.pmf(vals), "o", **kwargs)
            else:
                vals = np.linspace(low, high, 500)
                plt.plot(vals, self.rv.pdf(vals), **kwargs)
        plt.xlim(low - (high - low) * 0.1, high + (high - low) * 0.1)

        if show:
            self.show()