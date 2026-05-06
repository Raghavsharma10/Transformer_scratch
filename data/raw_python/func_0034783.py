def merge_all_edges_between_two_vertices(self, vertex1, vertex2):
        """ Merges all edge between two supplied vertices into a single edge from a perspective of multi-color merging.

         Proxies a call to :meth:`BreakpointGraph._BreakpointGraph__merge_all_bgedges_between_two_vertices`

        :param vertex1: a first out of two vertices edges between which are to be merged together
        :type vertex1: any python hashable object. :class:`bg.vertex.BGVertex` is expected
        :param vertex2: a second out of two vertices edges between which are to be merged together
        :type vertex2: any python hashable object. :class:`bg.vertex.BGVertex` is expected
        :return: ``None``, performs inplace changes
        """
        self.__merge_all_bgedges_between_two_vertices(vertex1=vertex1, vertex2=vertex2)