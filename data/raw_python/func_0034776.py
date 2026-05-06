def split_bgedge(self, bgedge, guidance=None, sorted_guidance=False,
                     account_for_colors_multiplicity_in_guidance=True,
                     key=None):
        """ Splits a :class:`bg.edge.BGEdge` in current :class:`BreakpointGraph` most similar to supplied one (if no unique identifier ``key`` is provided) with respect to supplied guidance.

        Proxies a call to :meth:`BreakpointGraph._BreakpointGraph__split_bgedge` method.

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
        self.__split_bgedge(bgedge=bgedge, guidance=guidance, sorted_guidance=sorted_guidance,
                            account_for_colors_multiplicity_in_guidance=account_for_colors_multiplicity_in_guidance,
                            key=key)