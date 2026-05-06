def edges(self, nbunch=None, keys=False):
        """ Iterates over edges in current :class:`BreakpointGraph` instance.

        Proxies a call to :meth:`BreakpointGraph._BreakpointGraph__edges`.

        :param nbunch: a vertex to iterate over edges outgoing from, if not provided,iteration over all edges is performed.
        :type nbuch: any hashable python object
        :param keys: a flag to indicate if information about unique edge's ids has to be returned alongside with edge
        :type keys: ``Boolean``
        :return: generator over edges in current :class:`BreakpointGraph`
        :rtype: ``generator``
        """
        for entry in self.__edges(nbunch=nbunch, keys=keys):
            yield entry