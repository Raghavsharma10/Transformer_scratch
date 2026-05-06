def delete_bgedge(self, bgedge, key=None):
        """ Deletes a supplied :class:`bg.edge.BGEdge` from a perspective of multi-color substitution. If unique identifier ``key`` is not provided, most similar (from perspective of :meth:`bg.multicolor.Multicolor.similarity_score` result) edge between respective vertices is chosen for change.

        Proxies a call to :math:`BreakpointGraph._BreakpointGraph__delete_bgedge` method.

        :param bgedge: an edge to be deleted from a perspective of multi-color substitution
        :type bgedge: :class:`bg.edge.BGEdge`
        :param key: unique identifier of existing edges in current :class:`BreakpointGraph` instance to be changed
        :type: any python object. ``int`` is expected.
        :return: ``None``, performed inplace changes.
        """
        self.__delete_bgedge(bgedge=bgedge, key=key)