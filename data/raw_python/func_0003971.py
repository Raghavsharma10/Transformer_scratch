def full_match(self, other):
        """Find the mapping between vertex indexes in self and other.

           This also works on disconnected graphs. Derived classes should just
           implement get_vertex_string and get_edge_string to make this method
           aware of the different nature of certain vertices. In case molecules,
           this would make the algorithm sensitive to atom numbers etc.
        """
        # we need normalize subgraphs because these graphs are used as patterns.
        graphs0 = [
            self.get_subgraph(group, normalize=True)
            for group in self.independent_vertices
        ]
        graphs1 = [
            other.get_subgraph(group)
            for group in other.independent_vertices
        ]

        if len(graphs0) != len(graphs1):
            return

        matches = []

        for graph0 in graphs0:
            pattern = EqualPattern(graph0)
            found_match = False
            for i, graph1 in enumerate(graphs1):
                local_matches = list(GraphSearch(pattern)(graph1, one_match=True))
                if len(local_matches) == 1:
                    match = local_matches[0]
                    # we need to restore the relation between the normalized
                    # graph0 and its original indexes
                    old_to_new = OneToOne((
                        (j, i) for i, j
                        in enumerate(graph0._old_vertex_indexes)
                    ))
                    matches.append(match * old_to_new)
                    del graphs1[i]
                    found_match = True
                    break
            if not found_match:
                return

        result = OneToOne()
        for match in matches:
            result.add_relations(match.forward.items())
        return result