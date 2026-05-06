def remove_node_by_value(self, value):
        """
        Delete all nodes in ``self.node_list`` with the value ``value``.

        Args:
            value (Any): The value to find and delete owners of.

        Returns: None

        Example:
            >>> from blur.markov.node import Node
            >>> node_1 = Node('One')
            >>> graph = Graph([node_1])
            >>> graph.remove_node_by_value('One')
            >>> len(graph.node_list)
            0
        """
        self.node_list = [node for node in self.node_list
                          if node.value != value]
        # Remove links pointing to the deleted node
        for node in self.node_list:
            node.link_list = [link for link in node.link_list if
                              link.target.value != value]