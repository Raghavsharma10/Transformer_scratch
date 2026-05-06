def add_graph_to_namespace(self, graph):
        """
        Adds the variables name to the namespace of the local LISP code

        :param graph: the graph to add to the namespace
        :return: None
        """
        for node in graph.vs:
            attributes = node.attributes()
            self.namespace[node['name']] = attributes
        for node in graph.es:
            attributes = node.attributes()
            self.namespace[node['name']] = attributes