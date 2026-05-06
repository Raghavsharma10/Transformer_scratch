def plot_vectors(self, arrows=True):
        """
        Plot vectors of positional transition of LISA values
        within quadrant in scatterplot in a polar plot.

        Parameters
        ----------
        ax : Matplotlib Axes instance, optional
            If given, the figure will be created inside this axis.
            Default =None.
        arrows : boolean, optional
            If True show arrowheads of vectors. Default =True
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

        from splot.giddy import dynamic_lisa_vectors

        fig, ax = dynamic_lisa_vectors(self, arrows=arrows)
        return fig, ax