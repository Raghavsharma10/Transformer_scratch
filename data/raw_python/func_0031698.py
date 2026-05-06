def raster_plots(self, xlim=[0, 1000], markersize=1, alpha=1., marker='o'):
        """
        Pretty plot of the spiking output of each population as raster and rate.
        
        
        Parameters
        ----------
        xlim : list
            List of floats. Spike time interval, e.g., `[0., 1000.]`.
        markersize : float
            marker size for plot, see `matplotlib.pyplot.plot`
        alpha : float
            transparency for markers, see `matplotlib.pyplot.plot`
        marker : :mod:`A valid marker style <matplotlib.markers>`
        
        
        Returns
        -------
        fig : `matplotlib.figure.Figure` object
        
        """
        x, y = self.get_xy(xlim)

        fig = plt.figure()
        fig.subplots_adjust(left=0.12, hspace=0.15)

        ax0 = fig.add_subplot(211)

        self.plot_raster(ax0, xlim, x, y, markersize=markersize, alpha=alpha,
                         marker=marker)
        remove_axis_junk(ax0)
        ax0.set_title('spike raster')
        ax0.set_xlabel("")

        nrows = len(self.X)
        bottom = np.linspace(0.1, 0.45, nrows+1)[::-1][1:]
        thickn = np.abs(np.diff(bottom))[0]*0.9


        for i, layer in enumerate(self.X):
            ax1 = fig.add_axes([0.12, bottom[i], 0.78, thickn])

            self.plot_f_rate(ax1, layer, i, xlim, x, y, )

            if i == nrows-1:
                ax1.set_xlabel('time (ms)')
            else:
                ax1.set_xticklabels([])

            if i == 4:
                ax1.set_ylabel(r'population rates ($s^{-1}$)')

            if i == 0:
                ax1.set_title(r'population firing rates ($s^{-1}$)')
              
        return fig