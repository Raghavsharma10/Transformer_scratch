def get_bounds(self):
        """Get the mesh bounds

        Returns
        -------
        bounds : list
            A list of tuples of mesh bounds.
        """
        if self._vertices_indexed_by_faces is not None:
            v = self._vertices_indexed_by_faces
        elif self._vertices is not None:
            v = self._vertices
        else:
            return None
        bounds = [(v[:, ax].min(), v[:, ax].max()) for ax in range(v.shape[1])]
        return bounds