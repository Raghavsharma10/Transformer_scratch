def writeMesh(self, vertices, faces, normals, values,
                  name='', reshape_faces=True):
        """ Write the given mesh instance.
        """

        # Store properties
        self._hasNormals = normals is not None
        self._hasValues = values is not None
        self._hasFaces = faces is not None

        # Get faces and number of vertices
        if faces is None:
            faces = np.arange(len(vertices))
            reshape_faces = True

        if reshape_faces:
            Nfaces = faces.size // 3
            faces = faces.reshape((Nfaces, 3))
        else:
            is_triangular = np.array([len(f) == 3
                                      for f in faces])
            if not(np.all(is_triangular)):
                logger.warning('''Faces doesn't appear to be triangular,
                be advised the file cannot be read back in vispy''')
        # Number of vertices
        N = vertices.shape[0]

        # Get string with stats
        stats = []
        stats.append('%i vertices' % N)
        if self._hasValues:
            stats.append('%i texcords' % N)
        else:
            stats.append('no texcords')
        if self._hasNormals:
            stats.append('%i normals' % N)
        else:
            stats.append('no normals')
        stats.append('%i faces' % faces.shape[0])

        # Write header
        self.writeLine('# Wavefront OBJ file')
        self.writeLine('# Created by vispy.')
        self.writeLine('#')
        if name:
            self.writeLine('# object %s' % name)
        else:
            self.writeLine('# unnamed object')
        self.writeLine('# %s' % ', '.join(stats))
        self.writeLine('')

        # Write data
        if True:
            for i in range(N):
                self.writeTuple(vertices[i], 'v')
        if self._hasNormals:
            for i in range(N):
                self.writeTuple(normals[i], 'vn')
        if self._hasValues:
            for i in range(N):
                self.writeTuple(values[i], 'vt')
        if True:
            for i in range(faces.shape[0]):
                self.writeFace(faces[i])