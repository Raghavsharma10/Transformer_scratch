def get_figure(self, heatmap_kw=None, **kwargs):
        """Generate a plotly figure showing the matrix as a heatmap.

        This is a shortcut for ``ExpMatrix.get_heatmap(...).get_figure(...)``.

        See :func:`ExpHeatmap.get_figure` for keyword arguments.

        Parameters
        ----------
        heatmap_kw : dict or None
            If not None, dictionary containing keyword arguments to be passed
            to the `ExpHeatmap` constructor.

        Returns
        -------
        `plotly.graph_objs.Figure`
            The plotly figure.
        """
        if heatmap_kw is not None:
            assert isinstance(heatmap_kw, dict)

        if heatmap_kw is None:
            heatmap_kw = {}

        return self.get_heatmap(**heatmap_kw).get_figure(**kwargs)