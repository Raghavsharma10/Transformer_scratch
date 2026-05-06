def add_graph(
        self,
        y,
        x_label=None,
        y_label="",
        title="",
        x_run=None,
        y_run=None,
        svg_size_px=None,
        key_position="bottom right",
    ):
        """
		Add a new graph to the overlap report.

		Args:
			y (str): Value plotted on y-axis.
			x_label (str): Label on x-axis.
			y_label (str): Label on y-axis.
			title (str): Title of the plot.
			x_run ((float,float)): x-range.
			y_run ((int,int)): y-rang.
			svg_size_px ((int,int): Size of SVG image in pixels.
			key_position (str): GnuPlot position of the legend.
		"""

        if x_run is None:
            x_run = self.default_x_run
        if y_run is None:
            y_run = self.default_y_run
        if svg_size_px is None:
            svg_size_px = self.default_svg_size_px

        for panel in self.panels:
            x_run = self._load_x_run(x_run)
            y_run = self._load_y_run(y_run)
            svg_size_px = self._load_svg_size_px(svg_size_px)
            panel.add_graph(
                y=y,
                x_run=x_run,
                y_run=y_run,
                svg_size_px=svg_size_px,
                y_label=y_label,
                x_label=x_label if x_label is not None else self.default_x_label,
                title=title,
                key_position=key_position,
            )