def save(self):
        """Serialize this mesh to a string appropriate for disk storage

        Returns
        -------
        state : dict
            The state.
        """
        import pickle
        if self._faces is not None:
            names = ['_vertices', '_faces']
        else:
            names = ['_vertices_indexed_by_faces']

        if self._vertex_colors is not None:
            names.append('_vertex_colors')
        elif self._vertex_colors_indexed_by_faces is not None:
            names.append('_vertex_colors_indexed_by_faces')

        if self._face_colors is not None:
            names.append('_face_colors')
        elif self._face_colors_indexed_by_faces is not None:
            names.append('_face_colors_indexed_by_faces')

        state = dict([(n, getattr(self, n)) for n in names])
        return pickle.dumps(state)