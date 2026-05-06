def set_data(self, xs=None, ys=None, zs=None, colors=None):
        '''Update the mesh data.

        Parameters
        ----------
        xs : ndarray | None
            A 2d array of x coordinates for the vertices of the mesh.
        ys : ndarray | None
            A 2d array of y coordinates for the vertices of the mesh.
        zs : ndarray | None
            A 2d array of z coordinates for the vertices of the mesh.
        colors : ndarray | None
            The color at each point of the mesh. Must have shape
            (width, height, 4) or (width, height, 3) for rgba or rgb
            color definitions respectively.
        '''

        if xs is None:
            xs = self._xs
            self.__vertices = None

        if ys is None:
            ys = self._ys
            self.__vertices = None

        if zs is None:
            zs = self._zs
            self.__vertices = None

        if self.__vertices is None:
            vertices, indices = create_grid_mesh(xs, ys, zs)
            self._xs = xs
            self._ys = ys
            self._zs = zs

        if self.__vertices is None:
            vertices, indices = create_grid_mesh(self._xs, self._ys, self._zs)
            self.__meshdata.set_vertices(vertices)
            self.__meshdata.set_faces(indices)

        if colors is not None:
            self.__meshdata.set_vertex_colors(colors.reshape(
                colors.shape[0] * colors.shape[1], colors.shape[2]))

        MeshVisual.set_data(self, meshdata=self.__meshdata)