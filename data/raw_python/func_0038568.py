def successors(self, node, exclude_compressed=True):
        """
        Returns the list of successors of a given node

        Parameters
        ----------
        node : str
            The target node

        exclude_compressed : boolean
            If true, compressed nodes are excluded from the successors list

        Returns
        -------
        list
            List of successors nodes
        """
        succs = super(Graph, self).successors(node)
        if exclude_compressed:
            return [n for n in succs if not self.node[n].get('compressed', False)]
        else:
            return succs