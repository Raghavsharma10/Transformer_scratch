def vertices(self):
        """Return an array (Nf, 3) of vertices.

        If only faces exist, the function computes the vertices and
        returns them.
        If no vertices or faces are specified, the function returns None.
        """

        if self._faces is None:
            if self._vertices is None:
                return None
            self.triangulate()
        return self._vertices