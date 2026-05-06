def appendSubgraph(self, parent_name,  graph_name, graph):
        """Utility method to associate Subgraph Instance to Root Graph Instance.

        This utility method is for use in constructor of child classes for 
        associating a MuninGraph Subgraph instance with a Root Graph instance.
        
        @param parent_name: Root Graph Name
        @param graph_name:  Subgraph Name
        @param graph:       MuninGraph Instance

        """
        if not self.isMultigraph:
            raise AttributeError("Simple Munin Plugins cannot have subgraphs.")
        if self._graphDict.has_key(parent_name):
            if not self._subgraphDict.has_key(parent_name):
                self._subgraphDict[parent_name] = {}
                self._subgraphNames[parent_name] = []
            self._subgraphDict[parent_name][graph_name] = graph
            self._subgraphNames[parent_name].append(graph_name)
        else:
            raise AttributeError("Invalid parent graph name %s used for subgraph %s."
                % (parent_name,  graph_name))