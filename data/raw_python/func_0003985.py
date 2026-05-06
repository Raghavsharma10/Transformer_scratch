def compare(self, vertex0, vertex1, subject_graph):
        """Returns true when the two vertices are of the same kind"""
        return (
            self.pattern_graph.vertex_fingerprints[vertex0] ==
            subject_graph.vertex_fingerprints[vertex1]
        ).all()