def remove_node(self, node):
        """
        Remove a node from ``self.node_list`` and links pointing to it.

        If ``node`` is not in the graph, do nothing.

        Args:
            node (Node): The node to be removed

        Returns: None

        Example:
            >>> from blur.markov.node import Node
            >>> node_1 = Node('One')
            >>> graph = Graph([node_1])
            >>> graph.remove_node(node_1)
            >>> len(graph.node_list)
            0
        """
        if node not in self.node_list:
            return
        self.node_list.remove(node)
        # Remove links pointing to the deleted node
        for n in self.node_list:
            n.link_list = [link for link in n.link_list if
                           link.target != node]