def extend(self, colors):
        """Extend a ColorArray with new colors

        Parameters
        ----------
        colors : instance of ColorArray
            The new colors.
        """
        colors = ColorArray(colors)
        self._rgba = np.vstack((self._rgba, colors._rgba))
        return self