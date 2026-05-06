def set_data(self, x=None, y=None, z=None, colors=None):
        """Update the data in this surface plot.

        Parameters
        ----------
        x : ndarray | None
            1D array of values specifying the x positions of vertices in the
            grid. If None, values will be assumed to be integers.
        y : ndarray | None
            1D array of values specifying the x positions of vertices in the
            grid. If None, values will be assumed to be integers.
        z : ndarray
            2D array of height values for each grid vertex.
        colors : ndarray
            (width, height, 4) array of vertex colors.
        """
        if x is not None:
            if self._x is None or len(x) != len(self._x):
                self.__vertices = None
            self._x = x

        if y is not None:
            if self._y is None or len(y) != len(self._y):
                self.__vertices = None
            self._y = y

        if z is not None:
            if self._x is not None and z.shape[0] != len(self._x):
                raise TypeError('Z values must have shape (len(x), len(y))')
            if self._y is not None and z.shape[1] != len(self._y):
                raise TypeError('Z values must have shape (len(x), len(y))')
            self._z = z
            if (self.__vertices is not None and
                    self._z.shape != self.__vertices.shape[:2]):
                self.__vertices = None

        if self._z is None:
            return

        update_mesh = False
        new_vertices = False

        # Generate vertex and face array
        if self.__vertices is None:
            new_vertices = True
            self.__vertices = np.empty((self._z.shape[0], self._z.shape[1], 3),
                                       dtype=np.float32)
            self.generate_faces()
            self.__meshdata.set_faces(self.__faces)
            update_mesh = True

        # Copy x, y, z data into vertex array
        if new_vertices or x is not None:
            if x is None:
                if self._x is None:
                    x = np.arange(self._z.shape[0])
                else:
                    x = self._x
            self.__vertices[:, :, 0] = x.reshape(len(x), 1)
            update_mesh = True

        if new_vertices or y is not None:
            if y is None:
                if self._y is None:
                    y = np.arange(self._z.shape[1])
                else:
                    y = self._y
            self.__vertices[:, :, 1] = y.reshape(1, len(y))
            update_mesh = True

        if new_vertices or z is not None:
            self.__vertices[..., 2] = self._z
            update_mesh = True

        if colors is not None:
            self.__meshdata.set_vertex_colors(colors)
            update_mesh = True

        # Update MeshData
        if update_mesh:
            self.__meshdata.set_vertices(
                self.__vertices.reshape(self.__vertices.shape[0] *
                                        self.__vertices.shape[1], 3))
            MeshVisual.set_data(self, meshdata=self.__meshdata)