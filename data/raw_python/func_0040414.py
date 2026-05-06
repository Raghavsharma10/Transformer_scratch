def feather_links(self, factor=0.01, include_self=False):
        """
        Feather the links of connected nodes.

        Go through every node in the network and make it inherit the links
        of the other nodes it is connected to. Because the link weight sum
        for any given node can be very different within a graph, the weights
        of inherited links are made proportional to the sum weight of the
        parent nodes.

        Args:
            factor (float): multiplier of neighbor links
            include_self (bool): whether nodes can inherit links pointing
                to themselves

        Returns: None

        Example:
            >>> from blur.markov.node import Node
            >>> node_1 = Node('One')
            >>> node_2 = Node('Two')
            >>> node_1.add_link(node_2, 1)
            >>> node_2.add_link(node_1, 1)
            >>> graph = Graph([node_1, node_2])
            >>> for link in graph.node_list[0].link_list:
            ...     print('{} {}'.format(link.target.value, link.weight))
            Two 1
            >>> graph.feather_links(include_self=True)
            >>> for link in graph.node_list[0].link_list:
            ...     print('{} {}'.format(link.target.value, link.weight))
            Two 1
            One 0.01
        """
        def feather_node(node):
            node_weight_sum = sum(l.weight for l in node.link_list)
            # Iterate over a copy of the original link list since we will
            # need to refer to this while modifying node.link_list
            for original_link in node.link_list[:]:
                neighbor_node = original_link.target
                neighbor_weight = original_link.weight
                feather_weight = neighbor_weight / node_weight_sum
                neighbor_node_weight_sum = sum(l.weight for
                                               l in neighbor_node.link_list)
                # Iterate over the links belonging to the neighbor_node,
                # copying its links to ``node`` with proportional weights
                for neighbor_link in neighbor_node.link_list:
                    if (not include_self) and (neighbor_link.target == node):
                        continue
                    relative_link_weight = (neighbor_link.weight /
                                            neighbor_node_weight_sum)
                    feathered_link_weight = round((relative_link_weight *
                                                   feather_weight * factor), 2)
                    node.add_link(neighbor_link.target, feathered_link_weight)
        for n in self.node_list:
            feather_node(n)