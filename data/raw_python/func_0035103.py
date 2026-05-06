def set_colorbar(self, plot_type, **kwargs):
        """Setup colorbar for specific type of plot.

        Specify a plot type to customize its corresponding colorbar in the figure.

        See the ColorbarContainer class attributes for more specific explanations.

        Args:
            plot_type (str): Type of plot to adjust. e.g. `Ratio`
            label (str, optional): Label for colorbar. Default is None.
            label_fontsize (int, optional): Fontsize for colorbar label. Default is None.
            ticks_fontsize (int, optional): Fontsize for colorbar tick labels. Default is None.
            pos (int, optional): Set a position for colorbar based on defaults. Default is None.
            colorbar_axes (len-4 list of floats): List for custom axes placement of the colorbar.
                See fig.add_axes from matplotlib.
                url: https://matplotlib.org/2.0.0/api/figure_api.html

        Raises:
            UserWarning: User calls set_colorbar without supplying any Args.
                This will not stop the code.

        """
        prop_default = {
            'cbar_label': None,
            'cbar_ticks_fontsize': 15,
            'cbar_label_fontsize': 20,
            'cbar_axes': [],
            'cbar_ticks': [],
            'cbar_tick_labels': [],
            'cbar_pos': 'use_default',
            'cbar_label_pad': None,
        }

        for prop, default in prop_default.items():
            kwargs[prop] = kwargs.get(prop[5:], default)

            if prop[5:] in kwargs:
                del kwargs[prop[5:]]

        if 'colorbars' not in self.figure.__dict__:
            self.figure.colorbars = {}

        self.figure.colorbars[plot_type] = kwargs
        return