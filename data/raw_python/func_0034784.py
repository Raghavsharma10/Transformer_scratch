def merge_all_edges(self):
        """ Merges all edges in a current :class`BreakpointGraph` instance between same pairs of vertices into a single edge from a perspective of multi-color merging.

        Iterates over all possible pairs of vertices in current :class:`BreakpointGraph` and merges all edges between respective pairs.

        :return: ``None``, performs inplace changes
        """
        pairs_of_vetices = [(edge.vertex1, edge.vertex2) for edge in self.edges()]
        for v1, v2 in pairs_of_vetices:
            ############################################################################################################
            #
            # we iterate over all pairs of vertices in the given graph and merge edges between them
            #
            ############################################################################################################
            self.__merge_all_bgedges_between_two_vertices(vertex1=v1, vertex2=v2)