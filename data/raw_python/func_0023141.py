def get_vertices(self, indexed=None):
        """Get the vertices

        Parameters
        ----------
        indexed : str | None
            If Note, return an array (N,3) of the positions of vertices in
            the mesh. By default, each unique vertex appears only once.
            If indexed is 'faces', then the array will instead contain three
            vertices per face in the mesh (and a single vertex may appear more
            than once in the array).

        Returns
        -------
        vertices : ndarray
            The vertices.
        """
        if indexed is None:
            if (self._vertices is None and
                    self._vertices_indexed_by_faces is not None):
                self._compute_unindexed_vertices()
            return self._vertices
        elif indexed == 'faces':
            if (self._vertices_indexed_by_faces is None and
                    self._vertices is not None):
                self._vertices_indexed_by_faces = \
                    self._vertices[self.get_faces()]
            return self._vertices_indexed_by_faces
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")