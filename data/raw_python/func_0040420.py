def pick(self, starting_node=None):
        """
        Pick a node on the graph based on the links in a starting node.

        Additionally, set ``self.current_node`` to the newly picked node.

        * if ``starting_node`` is specified, start from there
        * if ``starting_node`` is ``None``, start from ``self.current_node``
        * if ``starting_node`` is ``None`` and ``self.current_node``
          is ``None``, pick a uniformally random node in ``self.node_list``

        Args:
            starting_node (Node): ``Node`` to pick from.

        Returns: Node

        Example:
            >>> from blur.markov.node import Node
            >>> node_1 = Node('One')
            >>> node_2 = Node('Two')
            >>> node_1.add_link(node_1, 5)
            >>> node_1.add_link(node_2, 2)
            >>> node_2.add_link(node_1, 1)
            >>> graph = Graph([node_1, node_2])
            >>> [graph.pick().get_value() for i in range(5)]   # doctest: +SKIP
            ['One', 'One', 'Two', 'One', 'One']
        """
        if starting_node is None:
            if self.current_node is None:
                random_node = random.choice(self.node_list)
                self.current_node = random_node
                return random_node
            else:
                starting_node = self.current_node
        # Use weighted_choice on start_node.link_list
        self.current_node = weighted_choice(
            [(link.target, link.weight) for link in starting_node.link_list])
        return self.current_node