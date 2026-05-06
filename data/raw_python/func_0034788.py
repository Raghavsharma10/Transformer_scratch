def apply_kbreak(self, kbreak, merge=True):
        """ Check validity of supplied k-break and then applies it to current :class:`BreakpointGraph`

        Only :class:`bg.kbreak.KBreak` (or its heirs) instances are allowed as ``kbreak`` argument.
        KBreak must correspond to the valid kbreak and, since some changes to its internals might have been done since its creation, a validity check in terms of starting/resulting edges is performed.
        All vertices in supplied KBreak (except for paired infinity vertices) must be present in current :class:`BreakpointGraph`.
        For all supplied pairs of vertices (except for paired infinity vertices), there must be edges between such pairs of vertices, at least one of which must contain a multicolor matching a multicolor of supplied kbreak.

        Edges of specified in kbreak multicolor are deleted between supplied pairs of vertices in kbreak.start_edges (except for paired infinity vertices).
        New edges of specified in kbreak multicolor are added between all pairs of vertices in kbreak.result_edges (except for paired infinity vertices).
        If after the kbreak application there is an infinity vertex, that now has no edges incident to it, it is deleted form the current :class:`BreakpointGraph`.

        :param kbreak: a k-break to be applied to current :class:`BreakpointGraph`
        :type kbreak: `bg.kbreak.KBreak`
        :param merge: a flag to indicate on how edges, that will be created by a k-break, will be added to current :class:`BreakpointGraph`
        :type merge: ``Boolean``
        :return: nothing, performs inplace changes
        :rtype: ``None``
        :raises: ``ValueError``, ``TypeError``
        """
        ############################################################################################################
        #
        # k-break must ba valid to be applied
        #
        ############################################################################################################
        vertices = {}
        edge_data = {}
        if not isinstance(kbreak, KBreak):
            raise TypeError("Only KBreak and derivatives are allowed as kbreak argument")
        if not KBreak.valid_kbreak_matchings(kbreak.start_edges, kbreak.result_edges):
            raise ValueError("Supplied KBreak is not valid form perspective of starting/resulting sets of vertices")
        for vertex1, vertex2 in kbreak.start_edges:

            if vertex1.is_infinity_vertex and vertex2.is_infinity_vertex:
                ############################################################################################################
                #
                # when we encounter a fully infinity edge (both vertices are infinity vertices)
                # we shall not check if they are present in the current graph, because hat portion of a kbreak is artificial
                #
                ############################################################################################################
                continue
            if vertex1 not in self.bg or vertex2 not in self.bg:
                raise ValueError("Supplied KBreak targets vertices (`{v1}` and `{v2}`) at least one of which "
                                 "does not exist in current BreakpointGraph"
                                 "".format(v1=vertex1.name, v2=vertex2.name))
        for vertex1, vertex2 in kbreak.start_edges:
            if vertex1.is_infinity_vertex and vertex2.is_infinity_vertex:
                continue
            for bgedge in self.__edges_between_two_vertices(vertex1=vertex1, vertex2=vertex2):
                ############################################################################################################
                #
                # at least one edge between supplied pair of vertices must contain a multicolor that is specified for the kbreak
                #
                ############################################################################################################
                if kbreak.multicolor <= bgedge.multicolor:
                    break
            else:
                raise ValueError("Some targeted by kbreak edge with specified multicolor does not exists")
        for vertex1, vertex2 in kbreak.start_edges:
            if vertex1.is_infinity_vertex and vertex2.is_infinity_vertex:
                continue
            v1 = self.__get_vertex_by_name(vertex_name=vertex1.name)
            vertices[v1] = v1
            v2 = self.__get_vertex_by_name(vertex_name=vertex2.name)
            vertices[v2] = v2
            bgedge = BGEdge(vertex1=v1, vertex2=v2, multicolor=kbreak.multicolor)
            candidate_data, candidate_id, candidate_score = self.__determine_most_suitable_edge_for_deletion(
                bgedge=bgedge)
            data = candidate_data["attr_dict"]["data"]
            edge_data[v1] = data
            edge_data[v2] = data
            self.__delete_bgedge(bgedge=bgedge, keep_vertices=True)
        for vertex_set in kbreak.start_edges:
            for vertex in vertex_set:
                if vertex.is_infinity_vertex and vertex in self.bg:
                    ############################################################################################################
                    #
                    # after the first portion of a kbreak is performed one must make sure we don't leave any infinity vertices
                    # that have edges going to them, as infinity vertex is a special artificial vertex
                    #  and it has meaning only if there are edges going to / from it
                    #
                    ############################################################################################################
                    if len(list(self.get_edges_by_vertex(vertex=vertex))) == 0:
                        self.bg.remove_node(vertex)
        for vertex1, vertex2 in kbreak.result_edges:
            if vertex1.is_infinity_vertex and vertex2.is_infinity_vertex:
                ############################################################################################################
                #
                # if we encounter a pair of infinity vertices in result edges set, we shall not add them
                # as at least a part of kbreak corresponded to fusion
                # and those infinity edges on their own won't have any meaning
                #
                ############################################################################################################
                continue
            origin = kbreak.data.get("origin", None)
            v1 = vertices.get(vertex1, vertex1)
            v2 = vertices.get(vertex2, vertex2)
            bg_edge = BGEdge(vertex1=v1, vertex2=v2, multicolor=kbreak.multicolor)
            if "origin" in bg_edge.data:
                bg_edge.data["origin"] = origin
            if kbreak.is_a_fusion:
                edge1_data = edge_data[v1]
                edge2_data = edge_data[v2]
                merged_edge_fragment_data = merge_fragment_edge_data(edge1_data["fragment"], edge2_data["fragment"])
                result_edge_data = {}
                recursive_dict_update(result_edge_data, edge1_data)
                recursive_dict_update(result_edge_data, edge2_data)
                recursive_dict_update(result_edge_data, {"fragment": merged_edge_fragment_data})
                recursive_dict_update(bg_edge.data, result_edge_data)
            self.__add_bgedge(bg_edge, merge=merge)