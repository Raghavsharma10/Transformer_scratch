def add_edge(self, vertex1, vertex2, multicolor, merge=True, data=None):
        """ Creates a new :class:`bg.edge.BGEdge` object from supplied information and adds it to current instance of :class:`BreakpointGraph`.

        Proxies a call to :meth:`BreakpointGraph._BreakpointGraph__add_bgedge` method.

        :param vertex1: first vertex instance out of two in current :class:`BreakpointGraph`
        :type vertex1: any hashable object
        :param vertex2: second vertex instance out of two in current :class:`BreakpointGraph`
        :type vertex2: any hashable object
        :param multicolor: an information about multi-colors of added edge
        :type multicolor: :class:`bg.multicolor.Multicolor`
        :param merge: a flag to merge supplied information from multi-color perspective into a first existing edge between two supplied vertices
        :type merge: ``Boolean``
        :return: ``None``, performs inplace changes
        """
        self.__add_bgedge(BGEdge(vertex1=vertex1, vertex2=vertex2, multicolor=multicolor, data=data), merge=merge)