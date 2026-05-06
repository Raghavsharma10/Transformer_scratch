def setGraphVal(self, graph_name, field_name, val):
        """Utility method to set Value for Field in Graph.
        
        The private method is for use in retrieveVals() method of child classes.
        
        @param graph_name: Graph Name
        @param field_name: Field Name.
        @param val:        Value for field.

        """
        graph = self._getGraph(graph_name, True)
        if graph.hasField(field_name):
            graph.setVal(field_name, val)
        else:
            raise AttributeError("Invalid field name %s for graph %s." 
                                 % (field_name, graph_name))