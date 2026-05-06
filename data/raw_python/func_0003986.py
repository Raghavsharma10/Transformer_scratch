def iter_initial_relations(self, subject_graph):
        """Iterate over all valid initial relations for a match"""
        vertex0 = 0
        for vertex1 in range(subject_graph.num_vertices):
            yield vertex0, vertex1