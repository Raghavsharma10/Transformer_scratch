def subGraphHasField(self, parent_name, graph_name, field_name):
        """Return true if subgraph with name graph_name with parent graph with
        name parent_name has field with name field_name.
        
        @param parent_name: Root Graph Name
        @param graph_name:  Subgraph Name
        @param field_name:  Field Name.
        @return:            Boolean
        
        """
        subgraph = self._getSubGraph(parent_name, graph_name, True)
        return subgraph.hasField(field_name)