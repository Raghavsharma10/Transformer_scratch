def get_face_normals(self, indexed=None):
        """Get face normals

        Parameters
        ----------
        indexed : str | None
            If None, return an array (Nf, 3) of normal vectors for each face.
            If 'faces', then instead return an indexed array (Nf, 3, 3)
            (this is just the same array with each vector copied three times).

        Returns
        -------
        normals : ndarray
            The normals.
        """
        if self._face_normals is None:
            v = self.get_vertices(indexed='faces')
            self._face_normals = np.cross(v[:, 1] - v[:, 0],
                                          v[:, 2] - v[:, 0])

        if indexed is None:
            return self._face_normals
        elif indexed == 'faces':
            if self._face_normals_indexed_by_faces is None:
                norms = np.empty((self._face_normals.shape[0], 3, 3),
                                 dtype=np.float32)
                norms[:] = self._face_normals[:, np.newaxis, :]
                self._face_normals_indexed_by_faces = norms
            return self._face_normals_indexed_by_faces
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")