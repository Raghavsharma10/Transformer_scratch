def find_link(self, target_node):
        """
        Find the link that points to ``target_node`` if it exists.

        If no link in ``self`` points to ``target_node``, return None

        Args:
            target_node (Node): The node to look for in ``self.link_list``

        Returns:
            Link: An existing link pointing to ``target_node`` if found

            None: If no such link exists

        Example:
            >>> node_1 = Node('One')
            >>> node_2 = Node('Two')
            >>> node_1.add_link(node_2, 1)
            >>> link_1 = node_1.link_list[0]
            >>> found_link = node_1.find_link(node_2)
            >>> found_link == link_1
            True
        """
        try:
            return next(l for l in self.link_list if l.target == target_node)
        except StopIteration:
            return None