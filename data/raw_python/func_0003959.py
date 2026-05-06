def vertex_fingerprints(self):
        """A fingerprint for each vertex

           The result is invariant under permutation of the vertex indexes.
           Vertices that are symmetrically equivalent will get the same
           fingerprint, e.g. the hydrogens in methane would get the same
           fingerprint.
        """
        return self.get_vertex_fingerprints(
            [self.get_vertex_string(i) for i in range(self.num_vertices)],
            [self.get_edge_string(i) for i in range(self.num_edges)],
        )