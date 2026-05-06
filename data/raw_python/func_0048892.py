def add(self, *nodes):
        """ Adds nodes as siblings
        :param nodes: GraphNode(s)
        """
        for node in nodes:
            node.set_parent(self)
            self.add_sibling(node)