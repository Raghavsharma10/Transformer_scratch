def remove_links_to_self(self):
        """
        Remove any link in ``self.link_list`` whose ``target`` is ``self``.

        Returns: None

        Example:
            >>> node_1 = Node('One')
            >>> node_1.add_link(node_1, 5)
            >>> node_1.remove_links_to_self()
            >>> len(node_1.link_list)
            0
        """
        self.link_list = [link for link in self.link_list if
                          link.target != self]