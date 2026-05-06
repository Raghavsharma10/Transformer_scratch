def get_subgraph(self, subvertices, normalize=False):
        """Creates a subgraph of the current graph

           See :meth:`molmod.graphs.Graph.get_subgraph` for more information.
        """
        graph = Graph.get_subgraph(self, subvertices, normalize)
        if normalize:
            new_numbers = self.numbers[graph._old_vertex_indexes] # vertices do change
        else:
            new_numbers = self.numbers # vertices don't change!
        if self.symbols is None:
            new_symbols = None
        elif normalize:
            new_symbols = tuple(self.symbols[i] for i in graph._old_vertex_indexes)
        else:
            new_symbols = self.symbols
        new_orders = self.orders[graph._old_edge_indexes]
        result = MolecularGraph(graph.edges, new_numbers, new_orders, new_symbols)
        if normalize:
            result._old_vertex_indexes = graph._old_vertex_indexes
        result._old_edge_indexes = graph._old_edge_indexes
        return result