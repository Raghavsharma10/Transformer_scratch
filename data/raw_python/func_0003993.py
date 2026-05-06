def _iter_matches(self, input_match, subject_graph, one_match, level=0):
        """Given an onset for a match, iterate over all completions of that match

           This iterator works recursively. At each level the match is extended
           with a new set of relations based on vertices in the pattern graph
           that are at a distances 'level' from the starting vertex
        """
        self.print_debug("ENTERING _ITER_MATCHES", 1)
        self.print_debug("input_match: %s" % input_match)
        # A) collect the new edges in the pattern graph and the subject graph
        # to extend the match.
        #
        # Note that the edges are ordered. edge[0] is always in the match.
        # edge[1] is never in the match. The constraints contain information
        # about the end points of edges0. It is a list of two-tuples where
        # (a, b) means that a and b must be connected.
        #
        # Second note: suffix 0 indicates the pattern graph and suffix 1
        # is used for the subject graph.
        edges0, constraints0 = self.pattern.get_new_edges(level)
        edges1 = input_match.get_new_edges(subject_graph)
        self.print_debug("edges0: %s" % edges0)
        self.print_debug("constraints0: %s" % constraints0)
        self.print_debug("edges1: %s" % edges1)

        # B) iterate over the sets of new relations: [(vertex0[i], vertex1[j]),
        # ...] that contain all endpoints of edges0, that satisfy the
        # constraints0 and where (vertex0[i], vertex1[j]) only occurs if these
        # are end points of a edge0 and edge1 whose starting points are already
        # in init_match. These conditions are implemented in an iterator as to
        # separate concerns. This iterator also calls the routines that check
        # whether vertex1[j] also satisfies additional conditions inherent
        # vertex0[i].
        inr = self._iter_new_relations(input_match, subject_graph, edges0,
                                       constraints0, edges1)
        for new_relations in inr:
            # for each set of new_relations, construct a next_match and recurse
            next_match = input_match.copy_with_new_relations(new_relations)
            if not self.pattern.check_next_match(next_match, new_relations, subject_graph, one_match):
                continue
            if self.pattern.complete(next_match, subject_graph):
                yield next_match
            else:
                for match in self._iter_matches(next_match, subject_graph, one_match, level+1):
                    yield match
        self.print_debug("LEAVING_ITER_MATCHES", -1)