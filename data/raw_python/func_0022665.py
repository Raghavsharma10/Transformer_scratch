def set_data(self, vertices=None, tris=None, data=None):
        """Set the data

        Parameters
        ----------
        vertices : ndarray, shape (Nv, 3) | None
            Vertex coordinates.
        tris : ndarray, shape (Nf, 3) | None
            Indices into the vertex array.
        data : ndarray, shape (Nv,) | None
            scalar at vertices
        """
        # modifier pour tenier compte des None self._recompute = True
        if data is not None:
            self._data = data
            self._need_recompute = True
        if vertices is not None:
            self._vertices = vertices
            self._need_recompute = True
        if tris is not None:
            self._tris = tris
            self._need_recompute = True
        self.update()