def __split_bgedge(self, bgedge, guidance=None, sorted_guidance=False,
                       account_for_colors_multiplicity_in_guidance=True, key=None):
        """ Splits a :class:`bg.edge.BGEdge` in current :class:`BreakpointGraph` most similar to supplied one (if no unique identifier ``key`` is provided) with respect to supplied guidance.

        If no unique identifier for edge to be changed is specified, edge to be split is determined by iterating over all edges between vertices in supplied :class:`bg.edge.BGEdge` instance and the edge with most similarity score to supplied one is chosen.
        Once the edge to be split is determined, split if performed form a perspective of :class:`bg.multicolor.Multicolor` split.
        The originally detected edge is deleted, and new edges containing information about multi-colors after splitting, are added to the current :class:`BreakpointGraph`.


        :param bgedge: an edge to find most "similar to" among existing edges for a split
        :type bgedge: :class:`bg.edge.BGEdge`
        :param guidance: a guidance for underlying :class:`bg.multicolor.Multicolor` object to be split
        :type guidance: iterable where each entry is iterable with colors entries
        :param duplication_splitting: flag (**not** currently implemented) for a splitting of color-based splitting to take into account multiplicity of respective colors
        :type duplication_splitting: ``Boolean``
        :param key: unique identifier of edge to be split
        :type key: any python object. ``int`` is expected
        :return: ``None``, performs inplace changes
        """
        candidate_id = None
        candidate_score = 0
        candidate_data = None
        if key is not None:
            new_multicolors = Multicolor.split_colors(
                multicolor=self.bg[bgedge.vertex1][bgedge.vertex2][key]["attr_dict"]["multicolor"],
                guidance=guidance, sorted_guidance=sorted_guidance,
                account_for_color_multiplicity_in_guidance=account_for_colors_multiplicity_in_guidance)
            self.__delete_bgedge(bgedge=BGEdge(vertex1=bgedge.vertex1, vertex2=bgedge.vertex2,
                                               multicolor=self.bg[bgedge.vertex1][bgedge.vertex2][key]["attr_dict"]["multicolor"]),
                                 key=key)
            for multicolor in new_multicolors:
                self.__add_bgedge(BGEdge(vertex1=bgedge.vertex1, vertex2=bgedge.vertex2, multicolor=multicolor),
                                  merge=False)
        else:
            for v1, v2, key, data in self.bg.edges(nbunch=bgedge.vertex1, data=True, keys=True):
                if v2 == bgedge.vertex2:
                    score = Multicolor.similarity_score(bgedge.multicolor, data["attr_dict"]["multicolor"])
                    if score > candidate_score:
                        candidate_id = key
                        candidate_data = data
                        candidate_score = score
            if candidate_data is not None:
                new_multicolors = Multicolor.split_colors(multicolor=candidate_data["attr_dict"]["multicolor"],
                                                          guidance=guidance, sorted_guidance=sorted_guidance,
                                                          account_for_color_multiplicity_in_guidance=account_for_colors_multiplicity_in_guidance)
                self.__delete_bgedge(bgedge=BGEdge(vertex1=bgedge.vertex1, vertex2=bgedge.vertex2,
                                                   multicolor=candidate_data["attr_dict"]["multicolor"]),
                                     key=candidate_id)
                for multicolor in new_multicolors:
                    self.__add_bgedge(BGEdge(vertex1=bgedge.vertex1, vertex2=bgedge.vertex2,
                                             multicolor=multicolor), merge=False)