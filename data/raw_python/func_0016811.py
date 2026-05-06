def encode(self, cube_dimensions):
        """
        Produces a numpy array of integers which encode
        the supplied cube dimensions.
        """
        return np.asarray([getattr(cube_dimensions[d], s)
            for d in self._dimensions
            for s in self._schema],
                dtype=np.int32)