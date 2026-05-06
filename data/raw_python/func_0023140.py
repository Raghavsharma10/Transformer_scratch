def set_faces(self, faces):
        """Set the faces

        Parameters
        ----------
        faces : ndarray
            (Nf, 3) array of faces. Each row in the array contains
            three indices into the vertex array, specifying the three corners
            of a triangular face.
        """
        self._faces = faces
        self._edges = None
        self._edges_indexed_by_faces = None
        self._vertex_faces = None
        self._vertices_indexed_by_faces = None
        self.reset_normals()
        self._vertex_colors_indexed_by_faces = None
        self._face_colors_indexed_by_faces = None