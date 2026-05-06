def _getMultigraphID(self, graph_name, subgraph_name=None):
        """Private method for generating Multigraph ID from graph name and 
        subgraph name. 
        
        @param  graph_name:     Graph Name.
        @param  subgraph_name:  Subgraph Name.
        @return:                Multigraph ID.
        
        """
        if self.isMultiInstance and self._instanceName is not None:
            if subgraph_name is None:
                return "%s_%s" % (graph_name, self._instanceName)
            else:
                return "%s_%s.%s_%s" % (graph_name, self._instanceName, 
                                        subgraph_name, self._instanceName)
        else:
            if subgraph_name is None:
                return graph_name
            else:
                return "%s.%s" % (graph_name, subgraph_name)