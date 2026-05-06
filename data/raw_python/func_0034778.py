def split_all_edges_between_two_vertices(self, vertex1, vertex2, guidance=None, sorted_guidance=False,
                                             account_for_colors_multiplicity_in_guidance=True):
        """ Splits all edges between two supplied vertices in current :class:`BreakpointGraph` instance with respect to the provided guidance.

        Proxies a call to :meth:`BreakpointGraph._BreakpointGraph__split_all_edges_between_two_vertices` method.

        :param vertex1: a first out of two vertices edges between which are to be split
        :type vertex1: any python hashable object. :class:`bg.vertex.BGVertex` is expected
        :param vertex2: a second out of two vertices edges between which are to be split
        :type vertex2: any python hashable object. :class:`bg.vertex.BGVertex` is expected
        :param guidance: a guidance for underlying :class:`bg.multicolor.Multicolor` objects to be split
        :type guidance: iterable where each entry is iterable with colors entries
        :return: ``None``, performs inplace changes
        """
        self.__split_all_edges_between_two_vertices(vertex1=vertex1, vertex2=vertex2, guidance=guidance,
                                                    sorted_guidance=sorted_guidance,
                                                    account_for_colors_multiplicity_in_guidance=account_for_colors_multiplicity_in_guidance)