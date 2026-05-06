def iter_initial_relations(self, subject_graph):
        """Iterate over all valid initial relations for a match"""
        vertex0 = self.start_vertex
        for vertex1 in range(subject_graph.num_vertices):
            if self.compare(vertex0, vertex1, subject_graph):
                yield vertex0, vertex1