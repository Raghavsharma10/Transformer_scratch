def plot_summary(self, parameters=None, truth=None, extents=None, display=False,
                     filename=None, chains=None, figsize=1.0, errorbar=False, include_truth_chain=True,
                     blind=None, watermark=None, extra_parameter_spacing=0.5,
                     vertical_spacing_ratio=1.0, show_names=True):  # pragma: no cover
        """ Plots parameter summaries

        This plot is more for a sanity or consistency check than for use with final results.
        Plotting this before plotting with :func:`plot` allows you to quickly see if the
        chains give well behaved distributions, or if certain parameters are suspect
        or require a greater burn in period.


        Parameters
        ----------
        parameters : list[str]|int, optional
            Specify a subset of parameters to plot. If not set, all parameters are plotted.
            If an integer is given, only the first so many parameters are plotted.
        truth : list[float]|list|list[float]|dict[str]|str, optional
            A list of truth values corresponding to parameters, or a dictionary of
            truth values keyed by the parameter. Each "truth value" can be either a float (will
            draw a vertical line), two floats (a shaded interval) or three floats (min, mean, max),
            which renders as a shaded interval with a line for the mean. Or, supply a string
            which matches a chain name, and the results for that chain will be used as the 'truth'
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
        figsize : float, optional
            Scale horizontal and vertical figure size.
        errorbar : bool, optional
            Whether to onle plot an error bar, instead of the marginalised distribution.
        include_truth_chain : bool, optional
            If you specify another chain as the truth chain, determine if it should still
            be plotted.
        blind : bool|string|list[string], optional
            Whether to blind axes values. Can be set to `True` to blind all parameters,
            or can pass in a string (or list of strings) which specify the parameters to blind.
        watermark : str, optional
            A watermark to add to the figure
        extra_parameter_spacing : float, optional
            Increase horizontal space for parameter values
        vertical_spacing_ratio : float, optional
            Increase vertical space for each model
        show_names : bool, optional
            Whether to show chain names or not. Defaults to `True`.

        Returns
        -------
        figure
            the matplotlib figure created

        """
        wide_extents = not errorbar
        chains, parameters, truth, extents, blind = self._sanitise(chains, parameters, truth, extents, blind=blind, wide_extents=wide_extents)

        all_names = [c.name for c in self.parent.chains]

        # Check if we're using a chain for truth values
        if isinstance(truth, str):
            assert truth in all_names, "Truth chain %s is not in the list of added chains %s" % (truth, all_names)
            if not include_truth_chain:
                chains = [c for c in chains if c.name != truth]
            truth = self.parent.analysis.get_summary(chains=truth, parameters=parameters)

        max_param = self._get_size_of_texts(parameters)
        fid_dpi = 65  # Seriously I have no idea what value this should be
        param_width = extra_parameter_spacing + max(0.5, max_param / fid_dpi)

        if show_names:
            max_model_name = self._get_size_of_texts([chain.name for chain in chains])
            model_width = 0.25 + (max_model_name / fid_dpi)
            gridspec_kw = {'width_ratios': [model_width] + [param_width] * len(parameters), 'height_ratios': [1] * len(chains)}
            ncols = 1 + len(parameters)
        else:
            model_width = 0
            gridspec_kw = {'width_ratios': [param_width] * len(parameters), 'height_ratios': [1] * len(chains)}
            ncols = len(parameters)

        top_spacing = 0.3
        bottom_spacing = 0.2
        row_height = (0.5 if not errorbar else 0.3) * vertical_spacing_ratio
        width = param_width * len(parameters) + model_width
        height = top_spacing + bottom_spacing + row_height * len(chains)
        top_ratio = 1 - (top_spacing / height)
        bottom_ratio = bottom_spacing / height

        figsize = (width * figsize, height * figsize)
        fig, axes = plt.subplots(nrows=len(chains), ncols=ncols, figsize=figsize, squeeze=False, gridspec_kw=gridspec_kw)
        fig.subplots_adjust(left=0.05, right=0.95, top=top_ratio, bottom=bottom_ratio, wspace=0.0, hspace=0.0)
        label_font_size = self.parent.config["label_font_size"]
        legend_color_text = self.parent.config["legend_color_text"]

        max_vals = {}
        for i, row in enumerate(axes):
            chain = chains[i]

            cs, ws, ps, = chain.chain, chain.weights, chain.parameters
            gs, ns = chain.grid, chain.name

            colour = chain.config["color"]

            # First one put name of model
            if show_names:
                ax_first = row[0]
                ax_first.set_axis_off()
                text_colour = "k" if not legend_color_text else colour
                ax_first.text(0, 0.5, ns, transform=ax_first.transAxes, fontsize=label_font_size, verticalalignment="center", color=text_colour, weight="medium")
                cols = row[1:]
            else:
                cols = row

            for ax, p in zip(cols, parameters):
                # Set up the frames
                if i > 0:
                    ax.spines['top'].set_visible(False)
                if i < (len(chains) - 1):
                    ax.spines['bottom'].set_visible(False)
                if i < (len(chains) - 1) or p in blind:
                    ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(extents[p])

                # Put title in
                if i == 0:
                    ax.set_title(r"$%s$" % p, fontsize=label_font_size)

                # Add truth values
                truth_value = truth.get(p)
                if truth_value is not None:
                    if isinstance(truth_value, float) or isinstance(truth_value, int):
                        truth_mean = truth_value
                        truth_min, truth_max = None, None
                    else:
                        if len(truth_value) == 1:
                            truth_mean = truth_value
                            truth_min, truth_max = None, None
                        elif len(truth_value) == 2:
                            truth_min, truth_max = truth_value
                            truth_mean = None
                        else:
                            truth_min, truth_mean, truth_max = truth_value
                    if truth_mean is not None:
                        ax.axvline(truth_mean, **self.parent.config_truth)
                    if truth_min is not None and truth_max is not None:
                        ax.axvspan(truth_min, truth_max, color=self.parent.config_truth["color"], alpha=0.15, lw=0)
                # Skip if this chain doesnt have the parameter
                if p not in ps:
                    continue

                # Plot the good stuff
                if errorbar:
                    fv = self.parent.analysis.get_parameter_summary(chain, p)
                    if fv[0] is not None and fv[2] is not None:
                        diff = np.abs(np.diff(fv))
                        ax.errorbar([fv[1]], 0, xerr=[[diff[0]], [diff[1]]], fmt='o', color=colour)
                else:
                    m = self._plot_bars(ax, p, chain)
                    if max_vals.get(p) is None or m > max_vals.get(p):
                        max_vals[p] = m

        for i, row in enumerate(axes):
            index = 1 if show_names else 0
            for ax, p in zip(row[index:], parameters):
                if not errorbar:
                    ax.set_ylim(0, 1.1 * max_vals[p])

        dpi = 300
        if watermark:
            ax = None
            self._add_watermark(fig, ax, figsize, watermark, dpi=dpi, size_scale=0.8)

        if filename is not None:
            if isinstance(filename, str):
                filename = [filename]
            for f in filename:
                self._save_fig(fig, f, dpi)
        if display:
            plt.show()

        return fig