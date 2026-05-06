def __get_edge_by_two_vertices(self, vertex1, vertex2, key=None):
        """ Returns an instance of :class:`bg.edge.BBGEdge` edge between to supplied vertices (if ``key`` is supplied, returns a :class:`bg.edge.BBGEdge` instance about specified edge).

        Checks that both specified vertices are in current :class:`BreakpointGraph` and then depending on ``key`` argument, creates a new :class:`bg.edge.BBGEdge` instance and incorporates respective multi-color information into it.

        :param vertex1: first vertex instance out of two in current :class:`BreakpointGraph`
        :type vertex1: any hashable object
        :param vertex2: second vertex instance out of two in current :class:`BreakpointGraph`
        :type vertex2: any hashable object
        :param key: unique identifier of edge of interested to be retrieved from current :class:`BreakpointGraph`
        :type key: any python object. ``None`` or ``int`` is expected
        :return: edge between two specified edges respecting a ``key`` argument.
        :rtype: :class:`bg.edge.BGEdge`
        """
        if vertex1 in self.bg and vertex2 in self.bg[vertex1]:
            if key is None:
                key = min(self.bg[vertex1][vertex2])
            return BGEdge(vertex1=vertex1, vertex2=vertex2,
                          multicolor=self.bg[vertex1][vertex2][key]["attr_dict"]["multicolor"],
                          data=self.bg[vertex1][vertex2][key]["attr_dict"]["data"])
        return None