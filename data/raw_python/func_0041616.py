def add_reciprocal_link(self, target, weight):
        """
        Add links pointing in either direction between ``self`` and ``target``.

        This creates a ``Link`` from ``self`` to ``target`` and a ``Link``
        from ``target`` to ``self`` of equal weight. If ``target`` is a list
        of ``Node`` 's, repeat this for each one.

        Args:
            target (Node or list[Node]):
            weight (int or float):

        Returns: None

        Example:
            >>> node_1 = Node('One')
            >>> node_2 = Node('Two')
            >>> node_1.add_reciprocal_link(node_2, 5)
            >>> new_link_1 = node_1.link_list[0]
            >>> new_link_2 = node_2.link_list[0]
            >>> print(new_link_1)
            node.Link instance pointing to node with value "Two" with weight 5
            >>> print(new_link_2)
            node.Link instance pointing to node with value "One" with weight 5
        """
        # Generalize ``target`` to a list
        if not isinstance(target, list):
            target_list = [target]
        else:
            target_list = target
        for t in target_list:
            self.add_link(t, weight)
            t.add_link(self, weight)