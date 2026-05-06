def load(cls, graph, element):
        """
        Instantiates the node in the graph if it's not already stored in the graph

        :param graph: The dependency graph this node is a member of
        :type graph: corenlp_xml.dependencies.DependencyGraph
        :param element: The lxml element wrapping the node
        :type element: lxml.ElementBase

        """
        node = graph.get_node_by_idx(id(element.get("idx")))
        if node is None:
            node = cls(graph, element)
            graph.register_node(node)
        return node