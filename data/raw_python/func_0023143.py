def set_vertices(self, verts=None, indexed=None, reset_normals=True):
        """Set the mesh vertices

        Parameters
        ----------
        verts : ndarray | None
            The array (Nv, 3) of vertex coordinates.
        indexed : str | None
            If indexed=='faces', then the data must have shape (Nf, 3, 3) and
            is assumed to be already indexed as a list of faces. This will
            cause any pre-existing normal vectors to be cleared unless
            reset_normals=False.
        reset_normals : bool
            If True, reset the normals.
        """
        if indexed is None:
            if verts is not None:
                self._vertices = verts
            self._vertices_indexed_by_faces = None
        elif indexed == 'faces':
            self._vertices = None
            if verts is not None:
                self._vertices_indexed_by_faces = verts
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

        if reset_normals:
            self.reset_normals()