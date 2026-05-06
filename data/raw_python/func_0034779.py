def split_all_edges(self, guidance=None, sorted_guidance=False, account_for_colors_multiplicity_in_guidance=True):
        """ Splits all edge in current :class:`BreakpointGraph` instance with respect to the provided guidance.

        Iterate over all possible distinct pairs of vertices in current :class:`BreakpointGraph` instance and splits all edges between such pairs with respect to provided guidance.

        :param guidance: a guidance for underlying :class:`bg.multicolor.Multicolor` objects to be split
        :type guidance: iterable where each entry is iterable with colors entries
        :return: ``None``, performs inplace changes
        """
        vertex_pairs = [(edge.vertex1, edge.vertex2) for edge in self.edges()]
        for v1, v2 in vertex_pairs:
            self.__split_all_edges_between_two_vertices(vertex1=v1, vertex2=v2, guidance=guidance,
                                                        sorted_guidance=sorted_guidance,
                                                        account_for_colors_multiplicity_in_guidance=account_for_colors_multiplicity_in_guidance)