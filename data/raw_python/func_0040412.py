def merge_nodes(self, keep_node, kill_node):
        """
        Merge two nodes in the graph.

        Takes two nodes and merges them together, merging their links by
        combining the two link lists and summing the weights of links which
        point to the same node.

        All links in the graph pointing to ``kill_node`` will be merged
        into ``keep_node``.

        Links belonging to ``kill_node`` which point to targets not in
        ``self.node_list`` will not be merged into ``keep_node``

        Args:
            keep_node (Node): node to be kept
            kill_node (Node): node to be deleted

        Returns: None

        Example:
            >>> from blur.markov.node import Node
            >>> node_1 = Node('One')
            >>> node_2 = Node('Two')
            >>> node_3 = Node('Three')
            >>> node_1.add_link(node_3, 7)
            >>> node_2.add_link(node_1, 1)
            >>> node_2.add_link(node_2, 3)
            >>> node_3.add_link(node_2, 5)
            >>> graph = Graph([node_1, node_2, node_3])
            >>> print([node.value for node in graph.node_list])
            ['One', 'Two', 'Three']
            >>> graph.merge_nodes(node_2, node_3)
            >>> print([node.value for node in graph.node_list])
            ['One', 'Two']
            >>> for link in graph.node_list[1].link_list:
            ...     print('{} {}'.format(link.target.value, link.weight))
            One 1
            Two 8
        """
        # Merge links from kill_node to keep_node
        for kill_link in kill_node.link_list:
            if kill_link.target in self.node_list:
                keep_node.add_link(kill_link.target, kill_link.weight)
        # Merge any links in the graph pointing to kill_node into links
        # pointing to keep_node
        for node in self.node_list:
            for link in node.link_list:
                if link.target == kill_node:
                    node.add_link(keep_node, link.weight)
                    break
        # Remove kill_node from the graph
        self.remove_node(kill_node)