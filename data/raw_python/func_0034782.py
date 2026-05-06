def __merge_all_bgedges_between_two_vertices(self, vertex1, vertex2):
        """ Merges all edge between two supplied vertices into a single edge from a perspective of multi-color merging.

        :param vertex1: a first out of two vertices edges between which are to be merged together
        :type vertex1: any python hashable object. :class:`bg.vertex.BGVertex` is expected
        :param vertex2: a second out of two vertices edges between which are to be merged together
        :type vertex2: any python hashable object. :class:`bg.vertex.BGVertex` is expected
        :return: ``None``, performs inplace changes
        """
        ############################################################################################################
        #
        # no actual merging is performed, but rather all edges between two vertices are deleted
        # and then added with a merge argument set to true
        #
        ############################################################################################################
        edges_multicolors = [deepcopy(data["attr_dict"]["multicolor"]) for v1, v2, data in
                             self.bg.edges(nbunch=vertex1, data=True) if v2 == vertex2]
        self.__delete_all_bgedges_between_two_vertices(vertex1=vertex1, vertex2=vertex2)
        for multicolor in edges_multicolors:
            self.__add_bgedge(BGEdge(vertex1=vertex1, vertex2=vertex2, multicolor=multicolor), merge=True)