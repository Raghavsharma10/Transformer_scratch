def set_data(self, pos=None, color=None, width=None, connect=None):
        """ Set the data used to draw this visual.

        Parameters
        ----------
        pos : array
            Array of shape (..., 2) or (..., 3) specifying vertex coordinates.
        color : Color, tuple, or array
            The color to use when drawing the line. If an array is given, it
            must be of shape (..., 4) and provide one rgba color per vertex.
        width:
            The width of the line in px. Line widths < 1 px will be rounded up
            to 1 px when using the 'gl' method.
        connect : str or array
            Determines which vertices are connected by lines.
            * "strip" causes the line to be drawn with each vertex
              connected to the next.
            * "segments" causes each pair of vertices to draw an
              independent line segment
            * int numpy arrays specify the exact set of segment pairs to
              connect.
            * bool numpy arrays specify which _adjacent_ pairs to connect.
        """
        if pos is not None:
            self._bounds = None
            self._pos = pos
            self._changed['pos'] = True

        if color is not None:
            self._color = color
            self._changed['color'] = True

        if width is not None:
            self._width = width
            self._changed['width'] = True

        if connect is not None:
            self._connect = connect
            self._changed['connect'] = True

        self.update()