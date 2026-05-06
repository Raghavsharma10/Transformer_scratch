def scale(self, scale, center=None):
        """
        Scale the matrix about a given origin.

        The scaling is applied *after* the transformations already present
        in the matrix.

        Parameters
        ----------
        scale : array-like
            Scale factors along x, y and z axes.
        center : array-like or None
            The x, y and z coordinates to scale around. If None,
            (0, 0, 0) will be used.
        """
        scale = transforms.scale(as_vec4(scale, default=(1, 1, 1, 1))[0, :3])
        if center is not None:
            center = as_vec4(center)[0, :3]
            scale = np.dot(np.dot(transforms.translate(-center), scale),
                           transforms.translate(center))
        self.matrix = np.dot(self.matrix, scale)