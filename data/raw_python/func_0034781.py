def delete_all_edges_between_two_vertices(self, vertex1, vertex2):
        """ Deletes all edges between two supplied vertices

        Proxies a call to :meth:`BreakpointGraph._BreakpointGraph__delete_all_bgedges_between_two_vertices` method.

        :param vertex1: a first out of two vertices edges between which are to be deleted
        :type vertex1: any python hashable object. :class:`bg.vertex.BGVertex` is expected
        :param vertex2: a second out of two vertices edges between which are to be deleted
        :type vertex2: any python hashable object. :class:`bg.vertex.BGVertex` is expected
        :return: ``None``, performs inplace changes
        """
        self.__delete_all_bgedges_between_two_vertices(vertex1=vertex1, vertex2=vertex2)