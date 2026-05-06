def substitute_namespace_into_graph(self, graph):
        """
        Creates a graph from the local namespace of the code (to be used after the execution of the code)

        :param graph: The graph to use as a recipient of the namespace
        :return: the updated graph
        """
        for key, value in self.namespace.items():
            try:
                nodes = graph.vs.select(name=key)
                for node in nodes:
                    for k, v in value.items():
                        node[k] = v
            except:
                pass
            try:
                nodes = graph.es.select(name=key)
                for node in nodes:
                    for k, v in value.items():
                        node[k] = v
            except:
                pass
        return graph