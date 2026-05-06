def _set_pattern_graph(self, pattern_graph):
        """Initialize the pattern_graph"""
        self.pattern_graph = pattern_graph
        self.level_edges = {}
        self.level_constraints = {}
        self.duplicate_checks = set([])
        if pattern_graph is None:
            return
        if len(pattern_graph.independent_vertices) != 1:
            raise ValueError("A pattern_graph must not be a disconnected "
                             "graph.")
        # A) the levels for the incremental pattern matching
        ibfe = self.pattern_graph.iter_breadth_first_edges(self.start_vertex)
        for edge, distance, constraint in ibfe:
            if constraint:
                l = self.level_constraints.setdefault(distance-1, [])
            else:
                l = self.level_edges.setdefault(distance, [])
            l.append(edge)
        #print "level_edges", self.level_edges
        #print "level_constraints", self.level_constraints
        # B) The comparisons the should be checked when one wants to avoid
        # symmetrically duplicate pattern matches
        if self.criteria_sets is not None:
            for cycles in pattern_graph.symmetry_cycles:
                if len(cycles) > 0:
                    self.duplicate_checks.add((cycles[0][0], cycles[0][1]))