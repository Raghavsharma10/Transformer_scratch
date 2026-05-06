def map(self, coords):
        """Map coordinates

        Parameters
        ----------
        coords : array-like
            Coordinates to map.

        Returns
        -------
        coords : ndarray
            Coordinates.
        """
        m = np.empty(coords.shape)
        m[:, :3] = (coords[:, :3] * self.scale[np.newaxis, :3] +
                    coords[:, 3:] * self.translate[np.newaxis, :3])
        m[:, 3] = coords[:, 3]
        return m