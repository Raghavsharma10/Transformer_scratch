def __update(self, breakpoint_graph, merge_edges=False):
        """ Updates a current :class`BreakpointGraph` object with information from a supplied :class`BreakpointGraph` instance.

        Depending of a ``merge_edges`` flag, while updating of a current :class`BreakpointGraph` object is occuring, edges between similar vertices can be merged to already existing ones.

        :param breakpoint_graph: a breakpoint graph to extract information from, which will be then added to the current
        :type breakpoint_graph: :class`BreakpointGraph`
        :param merge_edges: flag to indicate if edges to be added to current :class`BreakpointGraph` object are to be merged to already existing ones
        :type merge_edges: ``Boolean``
        :return: ``None``, performs inplace changes
        """
        for bgedge in breakpoint_graph.edges():
            self.__add_bgedge(bgedge=deepcopy(bgedge), merge=merge_edges)