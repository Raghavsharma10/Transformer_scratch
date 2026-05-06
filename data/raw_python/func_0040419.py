def has_node_with_value(self, value):
        """
        Whether any node in ``self.node_list`` has the value ``value``.

        Args:
            value (Any): The value to find in ``self.node_list``

        Returns: bool

        Example:
            >>> from blur.markov.node import Node
            >>> node_1 = Node('One')
            >>> graph = Graph([node_1])
            >>> graph.has_node_with_value('One')
            True
            >>> graph.has_node_with_value('Foo')
            False
        """
        for node in self.node_list:
            if node.value == value:
                return True
        else:
            return False