def finish(self):
        """ Converts gathere lists to numpy arrays and creates
        BaseMesh instance.
        """
        self._vertices = np.array(self._vertices, 'float32')
        if self._faces:
            self._faces = np.array(self._faces, 'uint32')
        else:
            # Use vertices only
            self._vertices = np.array(self._v, 'float32')
            self._faces = None
        if self._normals:
            self._normals = np.array(self._normals, 'float32')
        else:
            self._normals = self._calculate_normals()
        if self._texcords:
            self._texcords = np.array(self._texcords, 'float32')
        else:
            self._texcords = None

        return self._vertices, self._faces, self._normals, self._texcords