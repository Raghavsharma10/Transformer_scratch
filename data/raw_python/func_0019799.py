def graphHasField(self, graph_name, field_name):
        """Return true if graph with name graph_name has field with 
        name field_name.
        
        @param graph_name: Graph Name
        @param field_name: Field Name.
        @return: Boolean
        
        """
        graph = self._graphDict.get(graph_name, True)
        return graph.hasField(field_name)