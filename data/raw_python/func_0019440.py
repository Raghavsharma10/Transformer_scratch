def plot_walks(self, parameters=None, truth=None, extents=None, display=False,
                   filename=None, chains=None, convolve=None, figsize=None,
                   plot_weights=True, plot_posterior=True, log_weight=None):  # pragma: no cover
        """ Plots the chain walk; the parameter values as a function of step index.

        This plot is more for a sanity or consistency check than for use with final results.
        Plotting this before plotting with :func:`plot` allows you to quickly see if the
        chains are well behaved, or if certain parameters are suspect
        or require a greater burn in period.

        The desired outcome is to see an unchanging distribution along the x-axis of the plot.
        If there are obvious tails or features in the parameters, you probably want
        to investigate.

        Parameters
        ----------
        parameters : list[str]|int, optional
            Specify a subset of parameters to plot. If not set, all parameters are plotted.
            If an integer is given, only the first so many parameters are plotted.
        truth : list[float]|dict[str], optional
            A list of truth values corresponding to parameters, or a dictionary of
            truth values keyed by the parameter.
        extents : list[tuple]|dict[str], optional
            A list of two-tuples for plot extents per parameter, or a dictionary of
            extents keyed by the parameter.
        display : bool, optional
            If set, shows the plot using ``plt.show()``
        filename : str, optional
            If set, saves the figure to the filename
        chains : int|str, list[str|int], optional
            Used to specify which chain to show if more than one chain is loaded in.
            Can be an integer, specifying the
            chain index, or a str, specifying the chain name.
        convolve : int, optional
            If set, overplots a smoothed version of the steps using ``convolve`` as
            the width of the smoothing filter.
        figsize : tuple, optional
            If set, sets the created figure size.
        plot_weights : bool, optional
            If true, plots the weight if they are available
        plot_posterior : bool, optional
            If true, plots the log posterior if they are available
        log_weight : bool, optional
            Whether to display weights in log space or not. If None, the value is
            inferred by the mean weights of the plotted chains.

        Returns
        -------
        figure
            the matplotlib figure created

        """

        chains, parameters, truth, extents, _ = self._sanitise(chains, parameters, truth, extents)

        n = len(parameters)
        extra = 0
        if plot_weights:
            plot_weights = plot_weights and np.any([np.any(c.weights != 1.0) for c in chains])

        plot_posterior = plot_posterior and np.any([c.posterior is not None for c in chains])

        if plot_weights:
            extra += 1
        if plot_posterior:
            extra += 1

        if figsize is None:
            figsize = (8, 0.75 + (n + extra))

        fig, axes = plt.subplots(figsize=figsize, nrows=n + extra, squeeze=False, sharex=True)

        for i, axes_row in enumerate(axes):
            ax = axes_row[0]
            if i >= extra:
                p = parameters[i - n]
                for chain in chains:
                    if p in chain.parameters:
                        chain_row = chain.get_data(p)
                        self._plot_walk(ax, p, chain_row, extents=extents.get(p), convolve=convolve, color=chain.config["color"])
                if truth.get(p) is not None:
                    self._plot_walk_truth(ax, truth.get(p))
            else:
                if i == 0 and plot_posterior:
                    for chain in chains:
                        if chain.posterior is not None:
                            self._plot_walk(ax, "$\log(P)$", chain.posterior - chain.posterior.max(),
                                            convolve=convolve, color=chain.config["color"])
                else:
                    if log_weight is None:
                        log_weight = np.any([chain.weights.mean() < 0.1 for chain in chains])
                    if log_weight:
                        for chain in chains:
                            self._plot_walk(ax, r"$\log_{10}(w)$", np.log10(chain.weights),
                                            convolve=convolve, color=chain.config["color"])
                    else:
                        for chain in chains:
                            self._plot_walk(ax, "$w$", chain.weights,
                                            convolve=convolve, color=chain.config["color"])

        if filename is not None:
            if isinstance(filename, str):
                filename = [filename]
            for f in filename:
                self._save_fig(fig, f, 300)
        if display:
            plt.show()
        return fig