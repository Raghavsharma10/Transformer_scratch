def node_path(self, node):
        """Return two lists describing the path from this node to another

        Parameters
        ----------
        node : instance of Node
            The other node.

        Returns
        -------
        p1 : list
            First path (see below).
        p2 : list
            Second path (see below).

        Notes
        -----
        The first list starts with this node and ends with the common parent
        between the endpoint nodes. The second list contains the remainder of
        the path from the common parent to the specified ending node.
        
        For example, consider the following scenegraph::
        
            A --- B --- C --- D
                   \
                    --- E --- F
        
        Calling `D.node_path(F)` will return::
        
            ([D, C, B], [E, F])
        
        """
        p1 = self.parent_chain()
        p2 = node.parent_chain()
        cp = None
        for p in p1:
            if p in p2:
                cp = p
                break
        if cp is None:
            raise RuntimeError("No single-path common parent between nodes %s "
                               "and %s." % (self, node))
        
        p1 = p1[:p1.index(cp)+1]
        p2 = p2[:p2.index(cp)][::-1]
        return p1, p2