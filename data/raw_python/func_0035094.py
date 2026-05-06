def add_legend(self, labels=None, **kwargs):
        """Specify legend for a plot.

        Adds labels and basic legend specifications for specific plot.

        For the optional Args, refer to
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
        for more information.

        # TODO: Add legend capabilities for Loss/Gain plots. This is possible
            using the return_fig_ax kwarg in the main plotting function.

        Args:
            labels (list of str): String representing each item in plot that
                will be added to the legend.

        Keyword Arguments:
            loc (str, int, len-2 list of floats, optional): Location of
                legend. See matplotlib documentation for more detail.
                Default is None.
            bbox_to_anchor (2-tuple or 4-tuple of floats, optional): Specify
                position and size of legend box. 2-tuple will specify (x,y)
                coordinate of part of box specified with `loc` kwarg.
                4-tuple will specify (x, y, width, height). See matplotlib
                documentation for more detail.
                Default is None.
            size (float, optional): Set size of legend using call to `prop`
                dict in legend call. See matplotlib documentaiton for more
                detail. Default is None.
            ncol (int, optional): Number of columns in the legend.
            Note: Other kwargs are available. See:
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html

        """
        if 'size' in kwargs:
            if 'prop' not in kwargs:
                kwargs['prop'] = {'size': kwargs['size']}
            else:
                kwargs['prop']['size'] = kwargs['size']
            del kwargs['size']
        self.legend.add_legend = True
        self.legend.legend_labels = labels
        self.legend.legend_kwargs = kwargs
        return