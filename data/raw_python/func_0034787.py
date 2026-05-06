def update(self, breakpoint_graph, merge_edges=False):
        """ Updates a current :class`BreakpointGraph` object with information from a supplied :class`BreakpointGraph` instance.

        Proxoes a call to :meth:`BreakpointGraph._BreakpointGraph__update` method.

        :param breakpoint_graph: a breakpoint graph to extract information from, which will be then added to the current
        :type breakpoint_graph: :class:`BreakpointGraph`
        :param merge_edges: flag to indicate if edges to be added to current :class`BreakpointGraph` object are to be merged to already existing ones
        :type merge_edges: ``Boolean``
        :return: ``None``, performs inplace changes
        """
        self.__update(breakpoint_graph=breakpoint_graph,
                      merge_edges=merge_edges)