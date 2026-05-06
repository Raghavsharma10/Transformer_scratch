def getGraphFieldList(self, graph_name):
        """Returns list of names of fields for graph with name graph_name.
        
        @param graph_name: Graph Name
        @return:           List of field names for graph.
        
        """
        graph = self._getGraph(graph_name, True)
        return graph.getFieldList()