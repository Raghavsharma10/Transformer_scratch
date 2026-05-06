def getSubgraphList(self, parent_name):
        """Returns list of names of subgraphs for Root Graph with name parent_name.
        
        @param parent_name: Name of Root Graph.
        @return:            List of subgraph names.
        
        """
        if not self.isMultigraph:
            raise AttributeError("Simple Munin Plugins cannot have subgraphs.")
        if self._graphDict.has_key(parent_name):
            return self._subgraphNames[parent_name] or []
        else:
            raise AttributeError("Invalid parent graph name %s."
                                 % (parent_name,))