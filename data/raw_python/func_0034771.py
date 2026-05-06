def __delete_bgedge(self, bgedge, key=None, keep_vertices=False):
        """ Deletes a supplied :class:`bg.edge.BGEdge` from a perspective of multi-color substitution. If unique identifier ``key`` is not provided, most similar (from perspective of :meth:`bg.multicolor.Multicolor.similarity_score` result) edge between respective vertices is chosen for change.

        If no unique identifier for edge to be changed is specified, edge to be updated is determined by iterating over all edges between vertices in supplied :class:`bg.edge.BGEdge` instance and the edge with most similarity score to supplied one is chosen.
        Once the edge to be substituted from is determined, substitution if performed form a perspective of :class:`bg.multicolor.Multicolor` substitution.
        If after substitution the remaining multicolor of respective edge is empty, such edge is deleted form a perspective of MultiGraph edge deletion.

        :param bgedge: an edge to be deleted from a perspective of multi-color substitution
        :type bgedge: :class:`bg.edge.BGEdge`
        :param key: unique identifier of existing edges in current :class:`BreakpointGraph` instance to be changed
        :type: any python object. ``int`` is expected.
        :return: ``None``, performed inplace changes.
        """
        ############################################################################################################
        #
        # determines which edge to delete
        # candidate edges setup
        #
        ############################################################################################################

        if key is not None:
            ############################################################################################################
            #
            # is an edge specific key is provided, only edge with that key can undergo multicolor deletion
            # even if that edge is not the most suited to the edge to be deleted
            #
            ############################################################################################################
            self.bg[bgedge.vertex1][bgedge.vertex2][key]["attr_dict"]["multicolor"] -= bgedge.multicolor
            if len(self.bg[bgedge.vertex1][bgedge.vertex2][key]["attr_dict"]["multicolor"].multicolors) == 0:
                ############################################################################################################
                #
                # since edge deletion correspond to multicolor substitution one must make sure
                # that no edges with empty multicolor are left in the graph
                #
                ############################################################################################################
                self.bg.remove_edge(v=bgedge.vertex1, u=bgedge.vertex2, key=key)
                if keep_vertices:
                    self.bg.add_node(bgedge.vertex1)
                    self.bg.add_node(bgedge.vertex2)
        else:
            candidate_data, candidate_id, candidate_score = self.__determine_most_suitable_edge_for_deletion(bgedge)
            if candidate_data is not None:
                candidate_data["attr_dict"]["multicolor"] -= bgedge.multicolor
                if len(self.bg[bgedge.vertex1][bgedge.vertex2][candidate_id]["attr_dict"][
                           "multicolor"].multicolors) == 0:
                    self.bg.remove_edge(v=bgedge.vertex1, u=bgedge.vertex2, key=candidate_id)
                    if keep_vertices:
                        self.bg.add_node(bgedge.vertex1)
                        self.bg.add_node(bgedge.vertex2)
        self.cache_valid["overall_set_of_colors"] = False