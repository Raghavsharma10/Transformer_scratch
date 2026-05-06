def add_link_to_self(self, source, weight):
        """
        Create and add a ``Link`` from a source node to ``self``.

        Args:
            source (Node): The node that will own the new ``Link``
                pointing to ``self``
            weight (int or float): The weight of the newly created ``Link``

        Returns: None

        Example:
            >>> node_1 = Node('One')
            >>> node_2 = Node('Two')
            >>> node_1.add_link_to_self(node_2, 5)
            >>> new_link = node_2.link_list[0]
            >>> print('{} {}'.format(new_link.target.value, new_link.weight))
            One 5
            >>> print(new_link)
            node.Link instance pointing to node with value "One" with weight 5
        """
        # Generalize source to a list to simplify code
        if not isinstance(source, list):
            source = [source]
        for source_node in source:
            source_node.add_link(self, weight=weight)