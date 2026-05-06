def _getSubGraph(self, parent_name, graph_name, fail_noexist=False):
        """Private method for returning subgraph object with name graph_name 
        and parent graph with name parent_name. 
        
        @param  parent_name: Root Graph Name
        @param  graph_name:  Subgraph Name
        @param fail_noexist: If true throw exception if there is no subgraph 
                             with name graph_name.
        @return:             Graph Object or None
        """
        if not self.isMultigraph:
            raise AttributeError("Simple Munin Plugins cannot have subgraphs.")
        if self._graphDict.has_key(parent_name) is not None:
            subgraphs = self._subgraphDict.get(parent_name)
            if subgraphs is not None:
                subgraph = subgraphs.get(graph_name)
                if fail_noexist and subgraph is None:
                    raise AttributeError("Invalid subgraph name %s"
                                         "for graph %s."
                                         % (graph_name, parent_name))
                else:
                    return subgraph
            else:
                raise AttributeError("Parent graph %s has no subgraphs."
                                     % (parent_name,))
        else:
            raise AttributeError("Invalid parent graph name %s "
                                 "for subgraph %s."
                                 % (parent_name, graph_name))