def edges_between_two_vertices(self, vertex1, vertex2, keys=False):
        """ Iterates over edges between two supplied vertices in current :class:`BreakpointGraph`

        Proxies a call to :meth:`Breakpoint._Breakpoint__edges_between_two_vertices` method.

        :param vertex1: a first vertex out of two, edges of interest are incident to
        :type vertex1: any hashable object, :class:`bg.vertex.BGVertex` is expected
        :param vertex2: a second vertex out of two, edges of interest are incident to
        :type vertex2: any hashable object, :class:`bg.vertex.BGVertex` is expected
        :param keys: a flag to indicate if information about unique edge's ids has to be returned alongside with edge
        :type keys: ``Boolean``
        :return: generator over edges (tuples ``edge, edge_id`` if keys specified) between two supplied vertices in current :class:`BreakpointGraph` wrapped in :class:`bg.vertex.BGVertex`
        :rtype: ``generator``
        """
        for entry in self.__edges_between_two_vertices(vertex1=vertex1, vertex2=vertex2, keys=keys):
            yield entry