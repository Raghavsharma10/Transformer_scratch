def get_vertex_colors(self, indexed=None):
        """Get vertex colors

        Parameters
        ----------
        indexed : str | None
            If None, return an array (Nv, 4) of vertex colors.
            If indexed=='faces', then instead return an indexed array
            (Nf, 3, 4).

        Returns
        -------
        colors : ndarray
            The vertex colors.
        """
        if indexed is None:
            return self._vertex_colors
        elif indexed == 'faces':
            if self._vertex_colors_indexed_by_faces is None:
                self._vertex_colors_indexed_by_faces = \
                    self._vertex_colors[self.get_faces()]
            return self._vertex_colors_indexed_by_faces
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")