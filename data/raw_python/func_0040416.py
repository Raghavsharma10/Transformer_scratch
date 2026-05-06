def find_node_by_value(self, value):
        """
        Find and return a node in self.node_list with the value ``value``.

        If multiple nodes exist with the value ``value``,
        return the first one found.

        If no such node exists, this returns ``None``.

        Args:
            value (Any): The value of the node to find

        Returns:
            Node: A node with value ``value`` if it was found

            None: If no node exists with value ``value``

        Example:
            >>> from blur.markov.node import Node
            >>> node_1 = Node('One')
            >>> graph = Graph([node_1])
            >>> found_node = graph.find_node_by_value('One')
            >>> found_node == node_1
            True
        """
        try:
            return next(n for n in self.node_list if n.value == value)
        except StopIteration:
            return None