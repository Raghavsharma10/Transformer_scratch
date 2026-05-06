def split_edge(self, vertex1, vertex2, multicolor, guidance=None, sorted_guidance=False,
                   account_for_colors_multiplicity_in_guidance=True, key=None):
        """ Splits an edge in current :class:`BreakpointGraph` most similar to supplied data (if no unique identifier ``key`` is provided) with respect to supplied guidance.

        Proxies a call to :meth:`BreakpointGraph._BreakpointGraph__split_bgedge` method.

        :param vertex1: a first vertex out of two the edge to be split is incident to
        :type vertex1: any python hashable object. :class:`bg.vertex.BGVertex` is expected
        :param vertex2: a second vertex out of two the edge to be split is incident to
        :type vertex2: any python hashable object. :class:`bg.vertex.BGVertex` is expected
        :param multicolor: a multi-color to find most suitable edge to be split
        :type multicolor: :class:`bg.multicolor.Multicolor`
        :param duplication_splitting: flag (**not** currently implemented) for a splitting of color-based splitting to take into account multiplicity of respective colors
        :type duplication_splitting: ``Boolean``
        :param key: unique identifier of edge to be split
        :type key: any python object. ``int`` is expected
        :return: ``None``, performs inplace changes
        """
        self.__split_bgedge(bgedge=BGEdge(vertex1=vertex1, vertex2=vertex2, multicolor=multicolor), guidance=guidance,
                            sorted_guidance=sorted_guidance,
                            account_for_colors_multiplicity_in_guidance=account_for_colors_multiplicity_in_guidance,
                            key=key)