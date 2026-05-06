def add_chain(self, chain, parameters=None, name=None, weights=None, posterior=None, walkers=None,
                  grid=False, num_eff_data_points=None, num_free_params=None, color=None, linewidth=None,
                  linestyle=None, kde=None, shade=None, shade_alpha=None, power=None, marker_style=None, marker_size=None,
                  marker_alpha=None, plot_contour=None, plot_point=None, statistics=None, cloud=None,
                  shade_gradient=None, bar_shade=None, bins=None, smooth=None, color_params=None,
                  plot_color_params=None, cmap=None, num_cloud=None):
        """
        Add a chain to the consumer.

        Parameters
        ----------
        chain : str|ndarray|dict
            The chain to load. Normally a ``numpy.ndarray``. If a string is found, it
            interprets the string as a filename and attempts to load it in. If a ``dict``
            is passed in, it assumes the dict has keys of parameter names and values of
            an array of samples. Notice that using a dictionary puts the order of
            parameters in the output under the control of the python ``dict.keys()`` function.
            If you passed ``grid`` is set, you can pass in the parameter ranges in list form.
        parameters : list[str], optional
            A list of parameter names, one for each column (dimension) in the chain. This parameter
            should remain ``None`` if a dictionary is given as ``chain``, as the parameter names
            are taken from the dictionary keys.
        name : str, optional
            The name of the chain. Used when plotting multiple chains at once.
        weights : ndarray, optional
            If given, uses this array to weight the samples in chain
        posterior : ndarray, optional
            If given, records the log posterior for each sample in the chain
        walkers : int, optional
            How many walkers went into creating the chain. Each walker should
            contribute the same number of steps, and should appear in contiguous
            blocks in the final chain.
        grid : boolean, optional
            Whether the input is a flattened chain from a grid search instead of a Monte-Carlo
            chains. Note that when this is set, `walkers` should not be set, and `weights` should
            be set to the posterior evaluation for the grid point. **Be careful** when using
            a coarse grid of setting a high smoothing value, as this may oversmooth the posterior
            surface and give unreasonably large parameter bounds.
        num_eff_data_points : int|float, optional
            The number of effective (independent) data points used in the model fitting. Not required
            for plotting, but required if loading in multiple chains to perform model comparison.
        num_free_params : int, optional
            The number of degrees of freedom in your model. Not required for plotting, but required if
            loading in multiple chains to perform model comparison.    
        color : str(hex), optional
            Provide a colour for the chain. Can be used instead of calling `configure` for convenience.
        linewidth : float, optional
            Provide a line width to plot the contours. Can be used instead of calling `configure` for convenience.
        linestyle : str, optional
            Provide a line style to plot the contour. Can be used instead of calling `configure` for convenience.
        kde : bool|float, optional
            Set the `kde` value for this specific chain. Can be used instead of calling `configure` for convenience.
        shade : booloptional
            If set, overrides the default behaviour and plots filled contours or not. If a list of
            bools is passed, you can turn shading on or off for specific chains.
        shade_alpha : float, optional
            Filled contour alpha value. Can be used instead of calling `configure` for convenience.
        power : float, optional
            The power to raise the posterior surface to. Useful for inflating or deflating uncertainty for debugging.
        marker_style : str|, optional
            The marker style to use when plotting points. Defaults to `'.'`
        marker_size : numeric|, optional
            Size of markers, if plotted. Defaults to `4`.
        marker_alpha : numeric, optional
            The alpha values when plotting markers.
        plot_contour : bool, optional
            Whether to plot the whole contour (as opposed to a point). Defaults to true for less than
            25 concurrent chains.
        plot_point : bool, optional
            Whether to plot a maximum likelihood point. Defaults to true for more then 24 chains.
        statistics : string, optional
            Which sort of statistics to use. Defaults to `"max"` for maximum likelihood
            statistics. Other available options are `"mean"`, `"cumulative"`, `"max_symmetric"`,
            `"max_closest"` and `"max_central"`. In the
            very, very rare case you want to enable different statistics for different
            chains, you can pass in a list of strings.
        cloud : bool, optional
            If set, overrides the default behaviour and plots the cloud or not        shade_gradient :
        bar_shade : bool, optional
            If set to true, shades in confidence regions in under histogram. By default
            this happens if you less than 3 chains, but is disabled if you are comparing
            more chains. You can pass a list if you wish to shade some chains but not others.
        bins : int|float, optional
            The number of bins to use. By default uses :math:`\frac{\sqrt{n}}{10}`, where
            :math:`n` are the number of data points. Giving an integer will set the number
            of bins to the given value. Giving a float will scale the number of bins, such
            that giving ``bins=1.5`` will result in using :math:`\frac{1.5\sqrt{n}}{10}` bins.
            Note this parameter is most useful if `kde=False` is also passed, so you
            can actually see the bins and not a KDE.        smooth : 
        color_params : str, optional
            The name of the parameter to use for the colour scatter. Defaults to none, for no colour. If set
            to 'weights', 'log_weights', or 'posterior' (without the quotes), and that is not a parameter in the chain, 
            it will respectively  use the weights, log weights, or posterior, to colour the points.
        plot_color_params : bool, optional
            Whether or not the colour parameter should also be plotted as a posterior surface.
        cmaps : str, optional
            The matplotlib colourmap to use in the `colour_param`. If you have multiple `color_param`s, you can
            specific a different cmap for each variable. By default ChainConsumer will cycle between several
            cmaps.
        num_cloud : int, optional
            The number of scatter points to show when enabling `cloud` or setting one of the parameters
            to colour scatter. Defaults to 15k per chain.
            
        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        is_dict = False
        assert chain is not None, "You cannot have a chain of None"
        if isinstance(chain, str):
            if chain.endswith("txt"):
                chain = np.loadtxt(chain)
            else:
                chain = np.load(chain)
        elif isinstance(chain, dict):
            assert parameters is None, \
                "You cannot pass a dictionary and specify parameter names"
            is_dict = True
            parameters = list(chain.keys())
            chain = np.array([chain[p] for p in parameters]).T
        elif isinstance(chain, list):
            chain = np.array(chain).T

        if grid:
            assert walkers is None, "If grid is set, walkers should not be"
            assert weights is not None, "If grid is set, you need to supply weights"
            if len(weights.shape) > 1:
                assert not is_dict, "We cannot construct a meshgrid from a dictionary, as the parameters" \
                                    "are no longer ordered. Please pass in a flattened array instead."
                self._logger.info("Constructing meshgrid for grid results")
                meshes = np.meshgrid(*[u for u in chain.T], indexing="ij")
                chain = np.vstack([m.flatten() for m in meshes]).T
                weights = weights.flatten()
                assert weights.size == chain[:,
                                       0].size, "Error, given weight array size disagrees with parameter sampling"

        if len(chain.shape) == 1:
            chain = chain[None].T

        if name is None:
            name = "Chain %d" % len(self.chains)

        if power is not None:
            assert isinstance(power, int) or isinstance(power, float), "Power should be numeric, but is %s" % type(
                power)

        if self._default_parameters is None and parameters is not None:
            self._default_parameters = parameters

        if parameters is None:
            if self._default_parameters is not None:
                assert chain.shape[1] == len(self._default_parameters), \
                    "Chain has %d dimensions, but default parameters have %d dimensions" \
                    % (chain.shape[1], len(self._default_parameters))
                parameters = self._default_parameters
                self._logger.debug("Adding chain using default parameters")
            else:
                self._logger.debug("Adding chain with no parameter names")
                parameters = ["%d" % x for x in range(chain.shape[1])]
        else:
            self._logger.debug("Adding chain with defined parameters")
            assert len(parameters) <= chain.shape[1], \
                "Have only %d columns in chain, but have been given %d parameters names! " \
                "Please double check this." % (chain.shape[1], len(parameters))
        for p in parameters:
            if p not in self._all_parameters:
                self._all_parameters.append(p)

        # Sorry, no KDE for you on a grid.
        if grid:
            kde = None
        if color is not None:
            color = self.color_finder.get_formatted([color])[0]

        c = Chain(chain, parameters, name, weights=weights, posterior=posterior, walkers=walkers,
                  grid=grid, num_free_params=num_free_params, num_eff_data_points=num_eff_data_points,
                  color=color, linewidth=linewidth, linestyle=linestyle, kde=kde, shade_alpha=shade_alpha, power=power,
                  marker_style=marker_style, marker_size=marker_size, marker_alpha=marker_alpha,
                  plot_contour=plot_contour, plot_point=plot_point, statistics=statistics, cloud=cloud,
                  shade=shade, shade_gradient=shade_gradient, bar_shade=bar_shade, bins=bins, smooth=smooth,
                  color_params=color_params, plot_color_params=plot_color_params, cmap=cmap,
                  num_cloud=num_cloud)
        self.chains.append(c)
        self._init_params()
        return self