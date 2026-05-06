def collate(self, graph):
        """
        :type graph: VariantGraph
        """
        # prepare the token index
        self.token_index.prepare()
        self.vertex_array = [None] * len(self.token_index.token_array)

        # Build the variant graph for the first witness
        # this is easy: generate a vertex for every token
        first_witness = self.collation.witnesses[0]
        tokens = first_witness.tokens()
        token_to_vertex = self.merge(graph, first_witness.sigil, tokens)
        # print("> token_to_vertex=", token_to_vertex)
        self.update_token_position_to_vertex(token_to_vertex)
        self.update_token_to_vertex_array(tokens, first_witness, self.token_position_to_vertex)

        # align witness 2 - n
        for x in range(1, len(self.collation.witnesses)):
            witness = self.collation.witnesses[x]
            tokens = witness.tokens()
            # print("\nwitness", witness.sigil)

            variant_graph_ranking = VariantGraphRanking.of(graph)
            # print("> x =", x, ", variant_graph_ranking =", variant_graph_ranking.byRank)
            variant_graph_ranks = list(set(map(lambda v: variant_graph_ranking.byVertex.get(v), graph.vertices())))
            # we leave in the rank of the start vertex, but remove the rank of the end vertex
            variant_graph_ranks.pop()

            # now the vertical stuff
            tokens_as_index_list = self.as_index_list(tokens)

            match_cube = MatchCube(self.token_index, witness, self.vertex_array, variant_graph_ranking,
                                   self.properties_filter)
            # print("> match_cube.matches=", match_cube.matches)
            self.fill_needleman_wunsch_table(variant_graph_ranks, tokens_as_index_list, match_cube)

            aligned = self.align_matching_tokens(match_cube)
            # print("> aligned=", aligned)
            # print("self.token_index.token_array=", self.token_index.token_array)
            # alignment = self.align_function(superbase, next_witness, token_to_vertex, match_cube)

            # merge
            witness_token_to_generated_vertex = self.merge(graph, witness.sigil, witness.tokens(), aligned)
            # print("> witness_token_to_generated_vertex =", witness_token_to_generated_vertex)
            token_to_vertex.update(witness_token_to_generated_vertex)
            # print("> token_to_vertex =", token_to_vertex)
            self.update_token_position_to_vertex(token_to_vertex, aligned)
            witness_token_position_to_vertex = {}
            for p in self.token_index.get_range_for_witness(witness.sigil):
                # print("> p= ", p)
                witness_token_position_to_vertex[p] = self.token_position_to_vertex[p]
            self.update_token_to_vertex_array(tokens, witness, witness_token_position_to_vertex)
            # print("> vertex_array =", self.vertex_array)

            #             print("actual")
            #             self._debug_edit_graph_table(self.table)
            #             print("expected")
            #             self._debug_edit_graph_table(self.table2)

            # change superbase
            # superbase = self.new_superbase

            if self.detect_transpositions:
                detector = TranspositionDetection(self)
                detector.detect()