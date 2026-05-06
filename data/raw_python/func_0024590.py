def add(self, node):
        """Add Node, replace existing node if node with node_id is present."""
        if not isinstance(node, Node):
            raise TypeError()
        for i, j in enumerate(self.__nodes):
            if j.node_id == node.node_id:
                self.__nodes[i] = node
                return
        self.__nodes.append(node)