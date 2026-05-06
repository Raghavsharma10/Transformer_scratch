def get_face_colors(self, indexed=None):
        """Get the face colors

        Parameters
        ----------
        indexed : str | None
            If indexed is None, return (Nf, 4) array of face colors.
            If indexed=='faces', then instead return an indexed array
            (Nf, 3, 4)  (note this is just the same array with each color
            repeated three times).
        
        Returns
        -------
        colors : ndarray
            The colors.
        """
        if indexed is None:
            return self._face_colors
        elif indexed == 'faces':
            if (self._face_colors_indexed_by_faces is None and
                    self._face_colors is not None):
                Nf = self._face_colors.shape[0]
                self._face_colors_indexed_by_faces = \
                    np.empty((Nf, 3, 4), dtype=self._face_colors.dtype)
                self._face_colors_indexed_by_faces[:] = \
                    self._face_colors.reshape(Nf, 1, 4)
            return self._face_colors_indexed_by_faces
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")