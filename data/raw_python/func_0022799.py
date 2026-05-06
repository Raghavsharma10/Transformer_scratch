def set_data(self, data=None, vertex_colors=None, face_colors=None,
                 color=None):
        """ Set the scalar array data

        Parameters
        ----------
        data : ndarray
            A 3D array of scalar values. The isosurface is constructed to show
            all locations in the scalar field equal to ``self.level``.
        vertex_colors : array-like | None
            Colors to use for each vertex.
        face_colors : array-like | None
            Colors to use for each face.
        color : instance of Color
            The color to use.
        """
        # We only change the internal variables if they are provided
        if data is not None:
            self._data = data
            self._recompute = True
        if vertex_colors is not None:
            self._vertex_colors = vertex_colors
            self._update_meshvisual = True
        if face_colors is not None:
            self._face_colors = face_colors
            self._update_meshvisual = True
        if color is not None:
            self._color = Color(color)
            self._update_meshvisual = True
        self.update()