def collate(self, graph, collation):
        '''
        :type graph: VariantGraph
        :type collation: Collation
        '''
        # Build the variant graph for the first witness
        # this is easy: generate a vertex for every token
        first_witness = collation.witnesses[0]
        tokens = first_witness.tokens()
        token_to_vertex = self.merge(graph, first_witness.sigil, tokens)

        # let the scorer prepare the first witness
        self.scorer.prepare_witness(first_witness)

        # construct superbase
        superbase = tokens

        # align witness 2 - n
        for x in range(1, len(collation.witnesses)):
            next_witness = collation.witnesses[x]

            # let the scorer prepare the next witness
            self.scorer.prepare_witness(next_witness)

            #             # VOOR CONTROLE!
            #             alignment = self._align_table(superbase, next_witness, token_to_vertex)
            #             self.table2 = self.table

            # alignment = token -> vertex
            alignment = self.align_function(superbase, next_witness, token_to_vertex)

            # merge
            token_to_vertex.update(self.merge(graph, next_witness.sigil, next_witness.tokens(), alignment))

            #             print("actual")
            #             self._debug_edit_graph_table(self.table)
            #             print("expected")
            #             self._debug_edit_graph_table(self.table2)

            # change superbase
            superbase = self.new_superbase

        if self.debug_scores:
            self._debug_edit_graph_table(self.table)