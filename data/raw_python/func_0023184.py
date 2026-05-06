def faces(self):
        """Return an array (Nf, 3) of vertex indexes, three per triangular
        face in the mesh.

        If faces have not been computed for this mesh, the function
        computes them.
        If no vertices or faces are specified, the function returns None.
        """

        if self._faces is None:
            if self._vertices is None:
                return None
            self.triangulate()
        return self._faces