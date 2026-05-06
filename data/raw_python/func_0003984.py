def iter_initial_relations(self, subject_graph):
        """Iterate over all valid initial relations for a match"""
        if self.pattern_graph.num_edges != subject_graph.num_edges:
            return # don't even try
        for pair in CustomPattern.iter_initial_relations(self, subject_graph):
            yield pair