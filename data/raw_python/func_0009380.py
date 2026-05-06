def plot(self, attribute=None, ax=None, **kwargs):
        """
        Plot the rose diagram.

        Parameters
        ----------
        attribute : (n,) ndarray, optional
            Variable to specify colors of the colorbars.
        ax : Matplotlib Axes instance, optional
            If given, the figure will be created inside this axis.
            Default =None. Note, this axis should have a polar projection.
        **kwargs : keyword arguments, optional
            Keywords used for creating and designing the plot.
            Note: 'c' and 'color' cannot be passed when attribute is not None

        Returns
        -------
        fig : Matplotlib Figure instance
            Moran scatterplot figure
        ax : matplotlib Axes instance
            Axes in which the figure is plotted

        """

        from splot.giddy import dynamic_lisa_rose
        fig, ax = dynamic_lisa_rose(self, attribute=attribute,
                                    ax=ax, **kwargs)
        return fig, ax