def matrix(self, _):
        """Return translation matrix in homogeneous coordinates."""
        x, y, z = self.coord
        return np.array([
            [1., 0., 0., x],
            [0., 1., 0., y],
            [0., 0., 1., z],
            [0., 0., 0., 1.]
        ])