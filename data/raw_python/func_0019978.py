def plot(self, hist=False, show=False, **kwargs):
        """
        Plot the distribution of the UncertainFunction. By default, the
        distribution is shown with a kernel density estimate (kde).
        
        Optional
        --------
        hist : bool
            If true, a density histogram is displayed (histtype='stepfilled')
        show : bool
            If ``True``, the figure will be displayed after plotting the 
            distribution. If ``False``, an explicit call to ``plt.show()`` is
            required to display the figure.
        kwargs : any valid matplotlib.pyplot.plot or .hist kwarg
        
        """
        import matplotlib.pyplot as plt

        vals = self._mcpts
        low = min(vals)
        high = max(vals)

        p = ss.kde.gaussian_kde(vals)
        xp = np.linspace(low, high, 100)

        if hist:
            h = plt.hist(
                vals,
                bins=int(np.sqrt(len(vals)) + 0.5),
                histtype="stepfilled",
                normed=True,
                **kwargs
            )
            plt.ylim(0, 1.1 * h[0].max())
        else:
            plt.plot(xp, p.evaluate(xp), **kwargs)

        plt.xlim(low - (high - low) * 0.1, high + (high - low) * 0.1)

        if show:
            self.show()