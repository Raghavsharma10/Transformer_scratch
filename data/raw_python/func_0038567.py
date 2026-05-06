def predecessors(self, node, exclude_compressed=True):
        """
        Returns the list of predecessors of a given node

        Parameters
        ----------
        node : str
            The target node

        exclude_compressed : boolean
            If true, compressed nodes are excluded from the predecessors list

        Returns
        -------
        list
            List of predecessors nodes
        """
        preds = super(Graph, self).predecessors(node)
        if exclude_compressed:
            return [n for n in preds if not self.node[n].get('compressed', False)]
        else:
            return preds