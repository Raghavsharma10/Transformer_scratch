def merge_links_from(self, other_node, merge_same_value_targets=False):
        """
        Merge links from another node with ``self.link_list``.

        Copy links from another node, merging when copied links point to a
        node which this already links to.

        Args:
            other_node (Node): The node to merge links from
            merge_same_value_targets (bool): Whether or not to merge links
                whose targets have the same value (but are not necessarily
                the same ``Node``). If False, links will only be merged
                when ``link_in_other.target == link_in_self.target``. If True,
                links will be merged when
                ``link_in_other.target.value == link_in_self.target.value``

        Returns: None

        Example:
            >>> node_1 = Node('One')
            >>> node_2 = Node('Two')
            >>> node_1.add_link(node_1, 1)
            >>> node_1.add_link(node_2, 3)
            >>> node_2.add_link(node_1, 4)
            >>> node_1.merge_links_from(node_2)
            >>> print(node_1)
            node.Node instance with value One with 2 links:
                0: 5 --> One
                1: 3 --> Two
        """
        for other_link in other_node.link_list:
            for existing_link in self.link_list:
                if merge_same_value_targets:
                    if other_link.target.value == existing_link.target.value:
                        existing_link.weight += other_link.weight
                        break
                else:
                    if other_link.target == existing_link.target:
                        existing_link.weight += other_link.weight
                        break
            else:
                self.add_link(other_link.target, other_link.weight)