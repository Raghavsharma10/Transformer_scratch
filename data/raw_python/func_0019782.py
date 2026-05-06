def _getGraph(self, graph_name, fail_noexist=False):
        """Private method for returning graph object with name graph_name. 
        
        @param graph_name:   Graph Name
        @param fail_noexist: If true throw exception if there is no graph with
                             name graph_name.
        @return:             Graph Object or None
        
        """
        graph = self._graphDict.get(graph_name)
        if fail_noexist and graph is None:
            raise AttributeError("Invalid graph name: %s" % graph_name)
        else:
            return graph