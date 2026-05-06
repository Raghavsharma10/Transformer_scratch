def set_data(self, pos=None, color=None):
        """Set the data

        Parameters
        ----------
        pos : float
            Position of the line along the axis.
        color : list, tuple, or array
            The color to use when drawing the line. If an array is given, it
            must be of shape (1, 4) and provide one rgba color per vertex.
        """
        if pos is not None:
            pos = float(pos)
            xy = self._pos
            if self._is_vertical:
                xy[0, 0] = pos
                xy[0, 1] = -1
                xy[1, 0] = pos
                xy[1, 1] = 1
            else:
                xy[0, 0] = -1
                xy[0, 1] = pos
                xy[1, 0] = 1
                xy[1, 1] = pos
            self._changed['pos'] = True

        if color is not None:
            color = np.array(color, dtype=np.float32)
            if color.ndim != 1 or color.shape[0] != 4:
                raise ValueError('color must be a 4 element float rgba tuple,'
                                 ' list or array')
            self._color = color
            self._changed['color'] = True