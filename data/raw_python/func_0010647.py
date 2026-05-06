def validate(self, graph):
        """ Validate the graph by checking whether it is a directed acyclic graph.

        Args:
            graph (DiGraph): Reference to a DiGraph object from NetworkX.

        Raises:
            DirectedAcyclicGraphInvalid: If the graph is not a valid dag.
        """
        if not nx.is_directed_acyclic_graph(graph):
            raise DirectedAcyclicGraphInvalid(graph_name=self._name)