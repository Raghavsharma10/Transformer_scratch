def valid_kbreak_matchings(start_edges, result_edges):
        """ A staticmethod check implementation that makes sure that degrees of vertices, that are affected by current :class:`KBreak`

        By the notion of k-break, it shall keep the degree of vertices in :class:`bg.breakpoint_graph.BreakpointGraph` the same, after its application.
        By utilizing the Counter class, such check is performed, as the number the vertex is mentioned corresponds to its degree.

        :param start_edges: a list of pairs of vertices, that specifies where edges shall be removed by :class:`KBreak`
        :type start_edges: ``list(tuple(vertex, vertex), ...)``
        :param result_edges: a list of pairs of vertices, that specifies where edges shall be created by :class:`KBreak`
        :type result_edges: ``list(tuple(vertex, vertex), ...)``
        :return: a flag indicating if the degree of vertices are equal in start / result edges, targeted by :class:`KBreak`
        :rtype: ``Boolean``
        """
        start_stats = Counter(vertex for vertex_pair in start_edges for vertex in vertex_pair)
        result_stats = Counter(vertex for vertex_pair in result_edges for vertex in vertex_pair)
        return start_stats == result_stats