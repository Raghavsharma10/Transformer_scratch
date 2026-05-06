def plot(self, data, color='k', symbol=None, line_kind='-', width=1.,
             marker_size=10., edge_color='k', face_color='b', edge_width=1.,
             title=None, xlabel=None, ylabel=None):
        """Plot a series of data using lines and markers

        Parameters
        ----------
        data : array | two arrays
            Arguments can be passed as ``(Y,)``, ``(X, Y)`` or
            ``np.array((X, Y))``.
        color : instance of Color
            Color of the line.
        symbol : str
            Marker symbol to use.
        line_kind : str
            Kind of line to draw. For now, only solid lines (``'-'``)
            are supported.
        width : float
            Line width.
        marker_size : float
            Marker size. If `size == 0` markers will not be shown.
        edge_color : instance of Color
            Color of the marker edge.
        face_color : instance of Color
            Color of the marker face.
        edge_width : float
            Edge width of the marker.
        title : str | None
            The title string to be displayed above the plot
        xlabel : str | None
            The label to display along the bottom axis
        ylabel : str | None
            The label to display along the left axis.

        Returns
        -------
        line : instance of LinePlot
            The line plot.

        See also
        --------
        marker_types, LinePlot
        """
        self._configure_2d()
        line = scene.LinePlot(data, connect='strip', color=color,
                              symbol=symbol, line_kind=line_kind,
                              width=width, marker_size=marker_size,
                              edge_color=edge_color,
                              face_color=face_color,
                              edge_width=edge_width)
        self.view.add(line)
        self.view.camera.set_range()
        self.visuals.append(line)

        if title is not None:
            self.title.text = title
        if xlabel is not None:
            self.xlabel.text = xlabel
        if ylabel is not None:
            self.ylabel.text = ylabel

        return line