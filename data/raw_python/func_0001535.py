def merge(self, graph, witness_sigil, witness_tokens, alignments={}):
        """
        :type graph: VariantGraph
        """
        # NOTE: token_to_vertex only contains newly generated vertices
        token_to_vertex = {}
        last = graph.start
        for token in witness_tokens:
            vertex = alignments.get(token, None)
            if not vertex:
                vertex = graph.add_vertex(token, witness_sigil)
                token_to_vertex[token] = vertex
            else:
                vertex.add_token(witness_sigil, token)
                # graph.add_token_to_vertex(vertex, token, witness_sigil)
            graph.connect(last, vertex, witness_sigil)
            last = vertex
        graph.connect(last, graph.end, witness_sigil)
        return token_to_vertex