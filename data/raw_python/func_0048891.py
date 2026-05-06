def depends(self, *nodes):
        """ Adds nodes as relatives to this one, and
        updates the relatives with self as children.
        :param nodes: GraphNode(s)
        """
        for node in nodes:
            self.add_relative(node)
            node.add_children(self)