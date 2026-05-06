def __delete_all_bgedges_between_two_vertices(self, vertex1, vertex2):
        """ Deletes all edges between two supplied vertices

        :param vertex1: a first out of two vertices edges between which are to be deleted
        :type vertex1: any python hashable object. :class:`bg.vertex.BGVertex` is expected
        :param vertex2: a second out of two vertices edges between which are to be deleted
        :type vertex2: any python hashable object. :class:`bg.vertex.BGVertex` is expected
        :return: ``None``, performs inplace changes
        """
        edges_to_be_deleted_with_keys = [(key, data) for v1, v2, key, data in self.bg.edges(nbunch=vertex1, keys=True,
                                                                                            data=True) if v2 == vertex2]
        for key, data in edges_to_be_deleted_with_keys:
            self.__delete_bgedge(BGEdge(vertex1=vertex1, vertex2=vertex2, multicolor=data["attr_dict"]["multicolor"]),
                                 key=key)