def appendGraph(self, graph_name, graph):
        """Utility method to associate Graph Object to Plugin.
        
        This utility method is for use in constructor of child classes for
        associating a MuninGraph instances to the plugin.
        
        @param graph_name:  Graph Name
        @param graph:       MuninGraph Instance

        """
        self._graphDict[graph_name] = graph
        self._graphNames.append(graph_name)
        if not self.isMultigraph  and len(self._graphNames) > 1:
            raise AttributeError("Simple Munin Plugins cannot have more than one graph.")