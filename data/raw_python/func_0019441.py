def plot_distributions(self, parameters=None, truth=None, extents=None, display=False,
                           filename=None, chains=None, col_wrap=4, figsize=None, blind=None):  # pragma: no cover
        """ Plots the 1D parameter distributions for verification purposes.

        This plot is more for a sanity or consistency check than for use with final results.
        Plotting this before plotting with :func:`plot` allows you to quickly see if the
        chains give well behaved distributions, or if certain parameters are suspect
        or require a greater burn in period.


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
        col_wrap : int, optional
            How many columns to plot before wrapping.
        figsize : tuple(float)|float, optional
            Either a tuple specifying the figure size or a float scaling factor.
        blind : bool|string|list[string], optional
            Whether to blind axes values. Can be set to `True` to blind all parameters,
            or can pass in a string (or list of strings) which specify the parameters to blind.

        Returns
        -------
        figure
            the matplotlib figure created

        """
        chains, parameters, truth, extents, blind = self._sanitise(chains, parameters, truth, extents, blind=blind)

        n = len(parameters)
        num_cols = min(n, col_wrap)
        num_rows = int(np.ceil(1.0 * n / col_wrap))

        if figsize is None:
            figsize = 1.0
        if isinstance(figsize, float):
            figsize_float = figsize
            figsize = (num_cols * 2 * figsize, num_rows * 2 * figsize)
        else:
            figsize_float = 1.0

        summary = self.parent.config["summary"]
        label_font_size = self.parent.config["label_font_size"]
        tick_font_size = self.parent.config["tick_font_size"]
        max_ticks = self.parent.config["max_ticks"]
        diagonal_tick_labels = self.parent.config["diagonal_tick_labels"]

        if summary is None:
            summary = len(self.parent.chains) == 1

        hspace = (0.8 if summary else 0.5) / figsize_float
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize, squeeze=False)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.05, hspace=hspace)

        formatter = ScalarFormatter(useOffset=False)
        formatter.set_powerlimits((-3, 4))

        for i, ax in enumerate(axes.flatten()):
            if i >= len(parameters):
                ax.set_axis_off()
                continue
            p = parameters[i]

            ax.set_yticks([])
            if p in blind:
                ax.set_xticks([])
            else:
                if diagonal_tick_labels:
                    _ = [l.set_rotation(45) for l in ax.get_xticklabels()]
                _ = [l.set_fontsize(tick_font_size) for l in ax.get_xticklabels()]
                ax.xaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                ax.xaxis.set_major_formatter(formatter)
            ax.set_xlim(extents.get(p) or self._get_parameter_extents(p, chains))
            max_val = None
            for chain in chains:
                if p in chain.parameters:
                    param_summary = summary and p not in blind
                    m = self._plot_bars(ax, p, chain, summary=param_summary)
                    if max_val is None or m > max_val:
                        max_val = m

            self._add_truth(ax, truth, None, py=p)
            ax.set_ylim(0, 1.1 * max_val)
            ax.set_xlabel(p, fontsize=label_font_size)

        if filename is not None:
            if isinstance(filename, str):
                filename = [filename]
            for f in filename:
                self._save_fig(fig, f, 300)
        if display:
            plt.show()
        return fig