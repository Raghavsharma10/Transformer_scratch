def readFace(self, line):
        """ Each face consists of three or more sets of indices. Each set
        consists of 1, 2 or 3 indices to vertices/normals/texcords.
        """

        # Get parts (skip first)
        indexSets = [num for num in line.split(' ') if num][1:]

        final_face = []
        for indexSet in indexSets:

            # Did we see this exact index earlier? If so, it's easy
            final_index = self._facemap.get(indexSet)
            if final_index is not None:
                final_face.append(final_index)
                continue

            # If not, we need to sync the vertices/normals/texcords ...

            # Get and store final index
            final_index = len(self._vertices)
            final_face.append(final_index)
            self._facemap[indexSet] = final_index

            # What indices were given?
            indices = [i for i in indexSet.split('/')]

            # Store new set of vertex/normal/texcords.
            # If there is a single face that does not specify the texcord
            # index, the texcords are ignored. Likewise for the normals.
            if True:
                vertex_index = self._absint(indices[0], len(self._v))
                self._vertices.append(self._v[vertex_index])
            if self._texcords is not None:
                if len(indices) > 1 and indices[1]:
                    texcord_index = self._absint(indices[1], len(self._vt))
                    self._texcords.append(self._vt[texcord_index])
                else:
                    if self._texcords:
                        logger.warning('Ignoring texture coordinates because '
                                       'it is not specified for all faces.')
                    self._texcords = None
            if self._normals is not None:
                if len(indices) > 2 and indices[2]:
                    normal_index = self._absint(indices[2], len(self._vn))
                    self._normals.append(self._vn[normal_index])
                else:
                    if self._normals:
                        logger.warning('Ignoring normals because it is not '
                                       'specified for all faces.')
                    self._normals = None

        # Check face
        if self._faces and len(self._faces[0]) != len(final_face):
            raise RuntimeError(
                'Vispy requires that all faces are either triangles or quads.')

        # Done
        return final_face