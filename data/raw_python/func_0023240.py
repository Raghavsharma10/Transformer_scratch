def set_data(self, pos=None, color=None):
        """Set the data

        Parameters
        ----------
        pos : list, tuple or numpy array
            Bounds of the region along the axis. len(pos) must be >=2.
        color : list, tuple, or array
            The color to use when drawing the line. It must have a shape of
            (1, 4) for a single color region or (len(pos), 4) for a multicolor
            region.
        """
        new_pos = self._pos
        new_color = self._color

        if pos is not None:
            num_elements = len(pos)
            pos = np.array(pos, dtype=np.float32)
            if pos.ndim != 1:
                raise ValueError('Expected 1D array')
            vertex = np.empty((num_elements * 2, 2), dtype=np.float32)
            if self._is_vertical:
                vertex[:, 0] = np.repeat(pos, 2)
                vertex[:, 1] = np.tile([-1, 1], num_elements)
            else:
                vertex[:, 1] = np.repeat(pos, 2)
                vertex[:, 0] = np.tile([1, -1], num_elements)
            new_pos = vertex
            self._changed['pos'] = True

        if color is not None:
            color = np.array(color, dtype=np.float32)
            num_elements = new_pos.shape[0] / 2
            if color.ndim == 2:
                if color.shape[0] != num_elements:
                    raise ValueError('Expected a color for each pos')
                if color.shape[1] != 4:
                    raise ValueError('Each color must be a RGBA array')
                color = np.repeat(color, 2, axis=0).astype(np.float32)
            elif color.ndim == 1:
                if color.shape[0] != 4:
                    raise ValueError('Each color must be a RGBA array')
                color = np.repeat([color], new_pos.shape[0], axis=0)
                color = color.astype(np.float32)
            else:
                raise ValueError('Expected a numpy array of shape '
                                 '(%d, 4) or (1, 4)' % num_elements)
            new_color = color
            self._changed['color'] = True

        # Ensure pos and color have the same size
        if new_pos.shape[0] != new_color.shape[0]:
            raise ValueError('pos and color does must have the same size')

        self._color = new_color
        self._pos = new_pos