def __add_bgedge(self, bgedge, merge=True):
        """ Adds supplied :class:`bg.edge.BGEdge` object to current instance of :class:`BreakpointGraph`.

        Checks that vertices in supplied :class:`bg.edge.BGEdge` instance actually are present in current :class:`BreakpointGraph` if **merge** option of provided. Otherwise a new edge is added to the current :class:`BreakpointGraph`.

        :param bgedge: instance of :class:`bg.edge.BGEdge` infromation form which is to be added to current :class:`BreakpointGraph`
        :type bgedge: :class:`bg.edge.BGEdge`
        :param merge: a flag to merge supplied information from multi-color perspective into a first existing edge between two supplied vertices
        :type merge: ``Boolean``
        :return: ``None``, performs inplace changes
        """
        if bgedge.vertex1 in self.bg and bgedge.vertex2 in self.bg[bgedge.vertex1] and merge:
            key = min(self.bg[bgedge.vertex1][bgedge.vertex2].keys())
            self.bg[bgedge.vertex1][bgedge.vertex2][key]["attr_dict"]["multicolor"] += bgedge.multicolor
            self.bg[bgedge.vertex1][bgedge.vertex2][key]["attr_dict"]["data"] = {}
        else:
            self.bg.add_edge(bgedge.vertex1, bgedge.vertex2, attr_dict={"multicolor": deepcopy(bgedge.multicolor),
                                                                        "data": bgedge.data})
        self.cache_valid["overall_set_of_colors"] = False