def get_vertex_normals(self, indexed=None):
        """Get vertex normals

        Parameters
        ----------
        indexed : str | None
            If None, return an (N, 3) array of normal vectors with one entry
            per unique vertex in the mesh. If indexed is 'faces', then the
            array will contain three normal vectors per face (and some
            vertices may be repeated).

        Returns
        -------
        normals : ndarray
            The normals.
        """
        if self._vertex_normals is None:
            faceNorms = self.get_face_normals()
            vertFaces = self.get_vertex_faces()
            self._vertex_normals = np.empty(self._vertices.shape,
                                            dtype=np.float32)
            for vindex in xrange(self._vertices.shape[0]):
                faces = vertFaces[vindex]
                if len(faces) == 0:
                    self._vertex_normals[vindex] = (0, 0, 0)
                    continue
                norms = faceNorms[faces]  # get all face normals
                norm = norms.sum(axis=0)  # sum normals
                renorm = (norm**2).sum()**0.5
                if renorm > 0:
                    norm /= renorm
                self._vertex_normals[vindex] = norm

        if indexed is None:
            return self._vertex_normals
        elif indexed == 'faces':
            return self._vertex_normals[self.get_faces()]
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")