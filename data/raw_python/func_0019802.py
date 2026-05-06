def getGraphFieldCount(self, graph_name):
        """Returns number of fields for graph with name graph_name.
        
        @param graph_name: Graph Name
        @return:           Number of fields for graph.
        
        """
        graph = self._getGraph(graph_name, True)
        return graph.getFieldCount()