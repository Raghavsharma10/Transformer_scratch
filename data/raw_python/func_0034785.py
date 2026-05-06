def merge(cls, breakpoint_graph1, breakpoint_graph2, merge_edges=False):
        """ Merges two given instances of :class`BreakpointGraph` into a new one, that gather all available information from both supplied objects.

        Depending of a ``merge_edges`` flag, while merging of two dat structures is occurring, edges between similar vertices can be merged during the creation of a result :class`BreakpointGraph` obejct.

        Accounts for subclassing.

        :param breakpoint_graph1: a first out of two :class`BreakpointGraph` instances to gather information from
        :type breakpoint_graph1: :class`BreakpointGraph`
        :param breakpoint_graph2: a second out of two :class`BreakpointGraph` instances to gather information from
        :type breakpoint_graph2: :class`BreakpointGraph`
        :param merge_edges: flag to indicate if edges in a new merged :class`BreakpointGraph` object has to be merged between same vertices, or if splitting from supplied graphs shall be preserved.
        :type merge_edges: ``Boolean``
        :return: a new breakpoint graph object that contains all information gathered from both supplied breakpoint graphs
        :rtype: :class`BreakpointGraph`
        """
        result = cls()
        for bgedge in breakpoint_graph1.edges():
            result.__add_bgedge(bgedge=bgedge, merge=merge_edges)
        for bgedge in breakpoint_graph2.edges():
            result.__add_bgedge(bgedge=bgedge, merge=merge_edges)
        return result