def set_face_colors(self, colors, indexed=None):
        """Set the face color array

        Parameters
        ----------
        colors : array
            Array of colors. Must have shape (Nf, 4) (indexed by face),
            or shape (Nf, 3, 4) (face colors indexed by faces).
        indexed : str | None
            Should be 'faces' if colors are indexed by faces.
        """
        colors = _fix_colors(colors)
        if colors.shape[0] != self.n_faces:
            raise ValueError('incorrect number of colors %s, expected %s'
                             % (colors.shape[0], self.n_faces))
        if indexed is None:
            if colors.ndim != 2:
                raise ValueError('colors must be 2D if indexed is None')
            self._face_colors = colors
            self._face_colors_indexed_by_faces = None
        elif indexed == 'faces':
            if colors.ndim != 3:
                raise ValueError('colors must be 3D if indexed is "faces"')
            self._face_colors = None
            self._face_colors_indexed_by_faces = colors
        else:
            raise ValueError('indexed must be None or "faces"')