def to_graph(self):
        """
        Converts the logical network to its underlying interaction graph

        Returns
        -------
        caspo.core.graph.Graph
            The underlying interaction graph
        """
        edges = set()
        for clause, target in self.edges_iter():
            for source, signature in clause:
                edges.add((source, target, signature))

        return Graph.from_tuples(edges)