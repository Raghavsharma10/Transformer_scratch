def setSubgraphVal(self,  parent_name,  graph_name, field_name, val):
        """Set Value for Field in Subgraph.

        The private method is for use in retrieveVals() method of child
        classes.
        
        @param parent_name: Root Graph Name
        @param graph_name:  Subgraph Name
        @param field_name:  Field Name.
        @param val:         Value for field.

        """
        subgraph = self._getSubGraph(parent_name, graph_name, True)
        if subgraph.hasField(field_name):
            subgraph.setVal(field_name, val)
        else:
            raise AttributeError("Invalid field name %s for subgraph %s "
                                 "of parent graph %s." 
                                 % (field_name, graph_name, parent_name))