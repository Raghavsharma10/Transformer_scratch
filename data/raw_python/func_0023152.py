def n_faces(self):
        """The number of faces in the mesh"""
        if self._faces is not None:
            return self._faces.shape[0]
        elif self._vertices_indexed_by_faces is not None:
            return self._vertices_indexed_by_faces.shape[0]