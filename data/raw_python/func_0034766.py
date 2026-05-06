def __get_edges_by_vertex(self, vertex, keys=False):
        """ Iterates over edges that are incident to supplied vertex argument in current :class:`BreakpointGraph`

        Checks that the supplied vertex argument exists in underlying MultiGraph object as a vertex, then iterates over all edges that are incident to it. Wraps each yielded object into :class:`bg.edge.BGEdge` object.

        :param vertex: a vertex object in current :class:`BreakpointGraph` object
        :type vertex: any hashable object. :class:`bg.vertex.BGVertex` object is expected.
        :param keys: a flag to indicate if information about unique edge's ids has to be returned alongside with edge
        :type keys: ``Boolean``
        :return: generator over edges (tuples ``edge, edge_id`` if keys specified) in current :class:`BreakpointGraph` wrapped in :class:`bg.vertex.BGEVertex`
        :rtype: ``generator``
        """
        if vertex in self.bg:
            for vertex2, edges in self.bg[vertex].items():
                for key, data in self.bg[vertex][vertex2].items():
                    bg_edge = BGEdge(vertex1=vertex, vertex2=vertex2, multicolor=data["attr_dict"]["multicolor"],
                                     data=data["attr_dict"]["data"])
                    if keys:
                        yield bg_edge, key
                    else:
                        yield bg_edge