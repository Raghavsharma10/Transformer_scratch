def delete_edge(self, vertex1, vertex2, multicolor, key=None):
        """ Creates a new :class:`bg.edge.BGEdge` instance from supplied information and deletes it from a perspective of multi-color substitution. If unique identifier ``key`` is not provided, most similar (from perspective of :meth:`bg.multicolor.Multicolor.similarity_score` result) edge between respective vertices is chosen for change.

        Proxies a call to :math:`BreakpointGraph._BreakpointGraph__delete_bgedge` method.

        :param vertex1: a first vertex out of two the edge to be deleted is incident to
        :type vertex1: any python hashable object. :class:`bg.vertex.BGVertex` is expected
        :param vertex2: a second vertex out of two the edge to be deleted is incident to
        :type vertex2: any python hashable object. :class:`bg.vertex.BGVertex` is expected
        :param multicolor: a multi-color to find most suitable edge to be deleted
        :type multicolor: :class:`bg.multicolor.Multicolor`
        :param key: unique identifier of existing edges in current :class:`BreakpointGraph` instance to be changed
        :type: any python object. ``int`` is expected.
        :return: ``None``, performed inplace changes.
        """
        self.__delete_bgedge(bgedge=BGEdge(vertex1=vertex1, vertex2=vertex2, multicolor=multicolor), key=key)