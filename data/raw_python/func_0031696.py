def plot_raster(self, ax, xlim, x, y, pop_names=False,
                    markersize=20., alpha=1., legend=True,
                    marker='o', rasterized=True):
        """
        Plot network raster plot in subplot object.
        
        
        Parameters
        ----------
        ax : `matplotlib.axes.AxesSubplot` object
            plot axes
        xlim : list
            List of floats. Spike time interval, e.g., [0., 1000.].
        x : dict
            Key-value entries are population name and neuron spike times.
        y : dict
            Key-value entries are population name and neuron gid number.
        pop_names: bool
            If True, show population names on yaxis instead of gid number.
        markersize : float
            raster plot marker size
        alpha : float in [0, 1]
            transparency of marker
        legend : bool
            Switch on axes legends.
        marker : str
            marker symbol for matplotlib.pyplot.plot
        rasterized : bool
            if True, the scatter plot will be treated as a bitmap embedded in
            pdf file output


        Returns
        -------
        None
        
        """
        yoffset = [sum(self.N_X) if X=='TC' else 0 for X in self.X]
        for i, X in enumerate(self.X):
            if y[X].size > 0:
                ax.plot(x[X], y[X]+yoffset[i], marker,
                    markersize=markersize,
                    mfc=self.colors[i],
                    mec='none' if marker in '.ov><v^1234sp*hHDd' else self.colors[i],
                    alpha=alpha,
                    label=X, rasterized=rasterized,
                    clip_on=True)
        
        #don't draw anything for the may-be-quiet TC population
        N_X_sum = 0
        for i, X in enumerate(self.X):
            if y[X].size > 0:
                N_X_sum += self.N_X[i]
        
        ax.axis([xlim[0], xlim[1],
                 self.GIDs[self.X[0]][0], self.GIDs[self.X[0]][0]+N_X_sum])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_ylabel('cell id', labelpad=0)
        ax.set_xlabel('$t$ (ms)', labelpad=0)
        if legend:
            ax.legend()
        if pop_names:
            yticks = []
            yticklabels = []
            for i, X in enumerate(self.X):
                if y[X] != []:
                    yticks.append(y[X].mean()+yoffset[i])
                    yticklabels.append(self.X[i])
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
        
        # Add some horizontal lines separating the populations
        for i, X in enumerate(self.X):
            if y[X].size > 0:
                ax.plot([xlim[0], xlim[1]],
                        [y[X].max()+yoffset[i], y[X].max()+yoffset[i]],
                        'k', lw=0.25)