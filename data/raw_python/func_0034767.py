def get_edges_by_vertex(self, vertex, keys=False):
        """ Iterates over edges that are incident to supplied vertex argument in current :class:`BreakpointGraph`

        Proxies a call to :meth:`Breakpoint._Breakpoint__get_edges_by_vertex` method.

        :param vertex: a vertex object in current :class:`BreakpointGraph` object
        :type vertex: any hashable object. :class:`bg.vertex.BGVertex` object is expected.
        :param keys: a flag to indicate if information about unique edge's ids has to be returned alongside with edge
        :type keys: ``Boolean``
        :return: generator over edges (tuples ``edge, edge_id`` if keys specified) in current :class:`BreakpointGraph` wrapped in :class:`bg.vertex.BGEVertex`
        :rtype: ``generator``
        """
        for entry in self.__get_edges_by_vertex(vertex=vertex, keys=keys):
            yield entry