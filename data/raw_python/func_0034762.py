def add_bgedge(self, bgedge, merge=True):
        """ Adds supplied :class:`bg.edge.BGEdge` object to current instance of :class:`BreakpointGraph`.

        Proxies a call to :meth:`BreakpointGraph._BreakpointGraph__add_bgedge` method.

        :param bgedge: instance of :class:`bg.edge.BGEdge` infromation form which is to be added to current :class:`BreakpointGraph`
        :type bgedge: :class:`bg.edge.BGEdge`
        :param merge: a flag to merge supplied information from multi-color perspective into a first existing edge between two supplied vertices
        :type merge: ``Boolean``
        :return: ``None``, performs inplace changes
        """
        self.__add_bgedge(bgedge=bgedge, merge=merge)