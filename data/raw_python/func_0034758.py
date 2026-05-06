def __edges(self, nbunch=None, keys=False):
        """ Iterates over edges in current :class:`BreakpointGraph` instance.

        Returns a generator over the edges in current :class:`BreakpointGraph` instance producing instances of :class:`bg.edge.BGEdge` instances wrapping around information in underlying MultiGraph object.

        :param nbunch: a vertex to iterate over edges outgoing from, if not provided,iteration over all edges is performed.
        :type nbuch: any hashable python object
        :param keys: a flag to indicate if information about unique edge's ids has to be returned alongside with edge
        :type keys: ``Boolean``
        :return: generator over edges in current :class:`BreakpointGraph`
        :rtype: ``generator``
        """
        for v1, v2, key, data in self.bg.edges(nbunch=nbunch, data=True, keys=True):
            bgedge = BGEdge(vertex1=v1, vertex2=v2, multicolor=data["attr_dict"]["multicolor"],
                            data=data["attr_dict"]["data"])
            if not keys:
                yield bgedge
            else:
                yield bgedge, key