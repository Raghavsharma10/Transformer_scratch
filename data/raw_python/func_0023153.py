def get_vertex_faces(self):
        """
        List mapping each vertex index to a list of face indices that use it.
        """
        if self._vertex_faces is None:
            self._vertex_faces = [[] for i in xrange(len(self.get_vertices()))]
            for i in xrange(self._faces.shape[0]):
                face = self._faces[i]
                for ind in face:
                    self._vertex_faces[ind].append(i)
        return self._vertex_faces