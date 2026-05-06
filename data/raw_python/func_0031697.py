def plot_f_rate(self, ax, X, i, xlim, x, y, binsize=1, yscale='linear',
                    plottype='fill_between', show_label=False, rasterized=False):
        """
        Plot network firing rate plot in subplot object.
        
        
        Parameters
        ----------
        ax : `matplotlib.axes.AxesSubplot` object.
        X : str
            Population name.
        i : int
            Population index in class attribute `X`.
        xlim : list of floats
            Spike time interval, e.g., [0., 1000.].
        x : dict
            Key-value entries are population name and neuron spike times.
        y : dict
            Key-value entries are population name and neuron gid number.
        yscale : 'str'
            Linear, log, or symlog y-axes in rate plot.
        plottype : str
            plot type string in `['fill_between', 'bar']`
        show_label : bool
            whether or not to show labels
        

        Returns
        -------
        None
        
        """
        
        bins = np.arange(xlim[0], xlim[1]+binsize, binsize)
        (hist, bins) = np.histogram(x[X], bins=bins)
        
        if plottype == 'fill_between':
            ax.fill_between(bins[:-1], hist * 1000. / self.N_X[i],
                    color=self.colors[i], lw=0.5, label=X, rasterized=rasterized,
                    clip_on=False)
            ax.plot(bins[:-1], hist * 1000. / self.N_X[i],
                    color='k', lw=0.5, label=X, rasterized=rasterized,
                    clip_on=False)
        elif plottype == 'bar':
            ax.bar(bins[:-1], hist * 1000. / self.N_X[i],
                    color=self.colors[i], label=X, rasterized=rasterized ,
                    linewidth=0.25, width=0.9, clip_on=False)
        else:
            mssg = "plottype={} not in ['fill_between', 'bar']".format(plottype)
            raise Exception(mssg)

        remove_axis_junk(ax)

        ax.axis(ax.axis('tight'))

        ax.set_yscale(yscale)

        ax.set_xlim(xlim[0], xlim[1])
        if show_label:
            ax.text(xlim[0] + .05*(xlim[1]-xlim[0]), ax.axis()[3]*1.5, X,
                    va='center', ha='left')