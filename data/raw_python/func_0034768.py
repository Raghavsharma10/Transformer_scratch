def __edges_between_two_vertices(self, vertex1, vertex2, keys=False):
        """ Iterates over edges between two supplied vertices in current :class:`BreakpointGraph`

        Checks that both supplied vertices are present in current breakpoint graph and then yield all edges that are located between two supplied vertices.
        If keys option is specified, then not just edges are yielded, but rather pairs (edge, edge_id) are yielded

        :param vertex1: a first vertex out of two, edges of interest are incident to
        :type vertex1: any hashable object, :class:`bg.vertex.BGVertex` is expected
        :param vertex2: a second vertex out of two, edges of interest are incident to
        :type vertex2: any hashable object, :class:`bg.vertex.BGVertex` is expected
        :param keys: a flag to indicate if information about unique edge's ids has to be returned alongside with edge
        :type keys: ``Boolean``
        :return: generator over edges (tuples ``edge, edge_id`` if keys specified) between two supplied vertices in current :class:`BreakpointGraph` wrapped in :class:`bg.vertex.BGVertex`
        :rtype: ``generator``
        """
        for vertex in vertex1, vertex2:
            if vertex not in self.bg:
                raise ValueError("Supplied vertex ({vertex_name}) is not present in current BreakpointGraph"
                                 "".format(vertex_name=str(vertex.name)))
        for bgedge, key in self.__get_edges_by_vertex(vertex=vertex1, keys=True):
            if bgedge.vertex2 == vertex2:
                if keys:
                    yield bgedge, key
                else:
                    yield bgedge