def getSubgraphFieldList(self, parent_name, graph_name):
        """Returns list of names of fields for subgraph with name graph_name
        and parent graph with name parent_name.
        
        @param parent_name: Root Graph Name
        @param graph_name:  Subgraph Name
        @return:            List of field names for subgraph.
        
        """
        graph = self._getSubGraph(parent_name, graph_name, True)
        return graph.getFieldList()