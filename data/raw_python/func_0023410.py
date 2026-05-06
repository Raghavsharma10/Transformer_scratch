def set_data(self, pos=None, color=None, width=None, connect=None,
                 arrows=None):
        """Set the data used for this visual

        Parameters
        ----------
        pos : array
            Array of shape (..., 2) or (..., 3) specifying vertex coordinates.
        color : Color, tuple, or array
            The color to use when drawing the line. If an array is given, it
            must be of shape (..., 4) and provide one rgba color per vertex.
            Can also be a colormap name, or appropriate `Function`.
        width:
            The width of the line in px. Line widths > 1px are only
            guaranteed to work when using 'agg' method.
        connect : str or array
            Determines which vertices are connected by lines.

                * "strip" causes the line to be drawn with each vertex
                  connected to the next.
                * "segments" causes each pair of vertices to draw an
                  independent line segment
                * numpy arrays specify the exact set of segment pairs to
                  connect.
        arrows : array
            A Nx4 matrix where each row contains the x and y coordinate of the
            first and second vertex of the arrow body. Remember that the second
            vertex is used as center point for the arrow head, and the first
            vertex is only used for determining the arrow head orientation.

        """

        if arrows is not None:
            self._arrows = arrows
            self._arrows_changed = True

        LineVisual.set_data(self, pos, color, width, connect)